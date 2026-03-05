#!/usr/bin/python

import sys
import os
import logging
import numpy as np
import pandas as pd

from sorcha.ephemeris.simulation_driver import create_ephemeris

from sorcha.modules.PPLinkingFilter import PPLinkingFilter
from sorcha.modules.PPTrailingLoss import PPTrailingLoss
from sorcha.modules.PPBrightLimit import PPBrightLimit
from sorcha.modules.PPCalculateApparentMagnitude import PPCalculateApparentMagnitude
from sorcha.modules.PPApplyFOVFilter import PPApplyFOVFilter
from sorcha.modules.PPSNRLimit import PPSNRLimit
from sorcha.modules import PPAddUncertainties, PPRandomizeMeasurements
from sorcha.modules import PPVignetting
from sorcha.modules.PPFadingFunctionFilter import PPFadingFunctionFilter
from sorcha.modules.PPFaintObjectCullingFilter import PPFaintObjectCullingFilter
from sorcha.modules.PPMatchPointingToObservations import PPMatchPointingToObservations
from sorcha.modules.PPMagnitudeLimit import PPMagnitudeLimit
from sorcha.modules.PPOutput import PPWriteOutput
from sorcha.modules.PPFootprintFilter import Footprint
from sorcha.modules.PPStats import stats

from sorcha.readers.CombinedDataReader import CombinedDataReader
from sorcha.readers.CSVReader import CSVDataReader
from sorcha.readers.EphemerisReader import EphemerisDataReader
from sorcha.readers.OrbitAuxReader import OrbitAuxReader


def _build_reader(args, sconfigs):
    """
    Build a CombinedDataReader with all aux and (optionally) ephemeris readers
    attached.  No subsetting is done here — workers call ``read_objects``
    directly on the individual sub-readers rather than using the positional
    ``read_rows`` / ``block_start`` mechanism.

    Parameters
    ----------
    args : sorchaArguments
    sconfigs : sorchaConfigs

    Returns
    -------
    reader : CombinedDataReader
    """
    pplogger = logging.getLogger(__name__)

    ephem_type = sconfigs.input.ephemerides_type
    ephem_primary = False
    reader = CombinedDataReader(ephem_primary=ephem_primary, verbose=False)

    # TODO: Once more ephemerides_types are added this should be wrapped in a EphemerisDataReader
    # That does the selection and checks. We are holding off adding this level of indirection until there
    # is a second ephemerides_type.

    if ephem_type.casefold() not in ["ar", "external"]:
        pplogger.error(f"PPReadAllInput: Unsupported value for ephemerides_type {ephem_type}")
        sys.exit(f"PPReadAllInput: Unsupported value for ephemerides_type {ephem_type}")
    if ephem_type.casefold() == "external":
        reader.add_ephem_reader(EphemerisDataReader(args.input_ephemeris_file, sconfigs.input.eph_format))

    reader.add_aux_data_reader(OrbitAuxReader(args.orbinfile, sconfigs.input.aux_format))
    reader.add_aux_data_reader(CSVDataReader(args.paramsinput, sconfigs.input.aux_format))
    if sconfigs.activity.comet_activity is not None or sconfigs.lightcurve.lc_model is not None:
        reader.add_aux_data_reader(CSVDataReader(args.complex_parameters, sconfigs.input.aux_format))

    for aux_reader in reader.aux_data_readers:
        aux_reader._build_id_map()

    return reader


def _read_chunk_by_ids(reader, obj_id_chunk, sconfigs):
    """
    Load a block of data for a specific list of object IDs by calling
    ``read_objects`` directly on the underlying sub-readers.

    This is used by worker processes, which already know *which* IDs they own
    and don't need the positional ``read_rows`` / ``block_start`` mechanism
    that ``read_block`` / ``read_aux_block`` use.

    Parameters
    ----------
    reader : CombinedDataReader
    obj_id_chunk : list
        Object IDs to load for this memory chunk.
    sconfigs : sorchaConfigs

    Returns
    -------
    observations or orbits_df : pd.DataFrame
        * For external ephemeris: a fully joined observations DataFrame
          (ephemeris + all aux data).
        * For AR ephemeris: an aux-only orbits DataFrame ready for
          ``create_ephemeris``.
    """
    ephem_type = sconfigs.input.ephemerides_type

    if ephem_type.casefold() == "external":
        # Load ephemeris rows for these IDs.
        ephem_df = reader.ephem_reader.read_objects(obj_id_chunk)

        # Join each aux reader onto the ephemeris frame.
        for aux_reader in reader.aux_data_readers:
            aux_df = aux_reader.read_objects(obj_id_chunk)
            ephem_df = ephem_df.join(aux_df.set_index("ObjID"), on="ObjID")

        return ephem_df

    else:  # AR — return aux data only; ephemeris is computed later
        primary_df = reader.aux_data_readers[0].read_objects(obj_id_chunk)

        for i, aux_reader in enumerate(reader.aux_data_readers):
            if i == 0:
                combined = primary_df
            else:
                aux_df = aux_reader.read_objects(obj_id_chunk)
                combined = combined.join(aux_df.set_index("ObjID"), on="ObjID")

        return combined


def _run_worker_chunk(worker_id, obj_id_subset, args, sconfigs, filterpointing):
    """
    Process a subset of objects through the full Sorcha pipeline.

    This function is designed to run inside a worker process spawned by
    ProcessPoolExecutor.  It mirrors the main processing loop in
    runLSSTSimulation but:
      * operates only on *obj_id_subset*
      * writes to a per-worker temporary output file
      * recreates any non-picklable objects (e.g. Footprint) locally

    Parameters
    ----------
    worker_id : int
        Index of this worker (used to name the temporary output file).
    obj_id_subset : list
        Object IDs that this worker is responsible for.
    args : sorchaArguments
        A *copy* of the top-level args with outfilestem already set to
        the worker-specific stem.
    sconfigs : sorchaConfigs
    filterpointing : pd.DataFrame

    Returns
    -------
    worker_outpath : str or None
        Full path to the temporary output file written by this worker, or
        None if no observations passed the filters.
    worker_stats_path : str or None
        Full path to the temporary stats file, or None.
    """
    pplogger = logging.getLogger(__name__)

    # Recreate footprint
    footprint = None
    if sconfigs.fov.camera_model == "footprint":
        footprint = Footprint(sconfigs.fov.footprint_path, args.surveyname)

    reader = _build_reader(args, sconfigs)

    # Split this worker's object IDs into memory-sized sub-chunks.
    # Workers use read_objects() directly -- NOT the positional read_rows /
    # block_start mechanism -- because they own a specific ID subset, not a
    # contiguous row range in the file.
    chunk_size = sconfigs.input.size_serial_chunk
    id_sub_chunks = [obj_id_subset[i : i + chunk_size] for i in range(0, len(obj_id_subset), chunk_size)]

    loopCounter = 0
    has_output = False

    for obj_id_chunk in id_sub_chunks:
        if sconfigs.input.ephemerides_type.casefold() == "external":
            observations = _read_chunk_by_ids(reader, obj_id_chunk, sconfigs)
        else:
            orbits_df = _read_chunk_by_ids(reader, obj_id_chunk, sconfigs)

            if not sconfigs.expert.brute_force:
                orbits_df = PPFaintObjectCullingFilter(
                    orbits_df,
                    filterpointing,
                    sconfigs.filters.mainfilter,
                    sconfigs.filters.observing_filters,
                    sconfigs.lightcurve.lc_model,
                    sconfigs.activity.comet_activity,
                )
                if len(orbits_df) == 0:
                    pplogger.info(
                        f"[worker {worker_id}] No objects pass faint-object culling for chunk "
                        f"{loopCounter}. Skipping."
                    )
                    loopCounter += 1
                    continue

            observations = create_ephemeris(orbits_df, filterpointing, args, sconfigs)

        if len(observations.index) == 0:
            pplogger.info(f"[worker {worker_id}] No ephemeris observations in chunk {loopCounter}. Skipping.")
            loopCounter += 1
            continue

        observations = PPMatchPointingToObservations(observations, filterpointing)

        observations = PPCalculateApparentMagnitude(
            observations,
            sconfigs.phasecurves.phase_function,
            sconfigs.filters.mainfilter,
            sconfigs.filters.othercolours,
            sconfigs.filters.observing_filters,
            sconfigs.activity.comet_activity,
            lightcurve_choice=sconfigs.lightcurve.lc_model,
            verbose=False,
        )

        if sconfigs.expert.trailing_losses_on:
            dmagDetect = PPTrailingLoss(observations, "circularPSF")
            observations["PSFMagTrue"] = dmagDetect + observations["trailedSourceMagTrue"]
        else:
            observations["PSFMagTrue"] = observations["trailedSourceMagTrue"]

        if sconfigs.expert.vignetting_on:
            observations["fiveSigmaDepth_mag"] = PPVignetting.vignettingEffects(observations)
        else:
            observations["fiveSigmaDepth_mag"] = observations["fieldFiveSigmaDepth_mag"]

        observations = PPAddUncertainties.addUncertainties(observations, sconfigs, args._rngs, verbose=False)

        if sconfigs.expert.randomization_on:
            observations = PPRandomizeMeasurements.randomizeAstrometryAndPhotometry(
                observations, sconfigs, args._rngs, verbose=False
            )
        else:
            observations["RATrue_deg"] = observations["RA_deg"].copy()
            observations["DecTrue_deg"] = observations["Dec_deg"].copy()
            observations["trailedSourceMag"] = observations["trailedSourceMagTrue"].copy()
            observations["PSFMag"] = observations["PSFMagTrue"].copy()

        if sconfigs.fov.camera_model != "none" and len(observations.index) > 0:
            observations = PPApplyFOVFilter(
                observations,
                sconfigs,
                args._rngs,
                visits=args.visits,
                footprint=footprint,
                verbose=False,
            )

        if sconfigs.expert.snr_limit_on and len(observations.index) > 0:
            observations = PPSNRLimit(observations, sconfigs.expert.snr_limit)

        if sconfigs.expert.mag_limit_on and len(observations.index) > 0:
            observations = PPMagnitudeLimit(observations, sconfigs.expert.mag_limit)

        if sconfigs.fadingfunction.fading_function_on and len(observations.index) > 0:
            observations = PPFadingFunctionFilter(
                observations,
                sconfigs.fadingfunction.fading_function_peak_efficiency,
                sconfigs.fadingfunction.fading_function_width,
                args._rngs,
                verbose=False,
            )

        if sconfigs.saturation.bright_limit_on and len(observations.index) > 0:
            observations = PPBrightLimit(
                observations, sconfigs.filters.observing_filters, sconfigs.saturation.bright_limit
            )

        if sconfigs.linkingfilter.ssp_linking_on and len(observations.index) > 0:
            observations = PPLinkingFilter(
                observations,
                sconfigs.linkingfilter.ssp_detection_efficiency,
                sconfigs.linkingfilter.ssp_number_observations,
                sconfigs.linkingfilter.ssp_number_tracklets,
                sconfigs.linkingfilter.ssp_track_window,
                sconfigs.linkingfilter.ssp_separation_threshold,
                sconfigs.linkingfilter.ssp_maximum_time,
                sconfigs.linkingfilter.ssp_night_start_utc,
                drop_unlinked=sconfigs.linkingfilter.drop_unlinked,
            )
            observations.reset_index(drop=True, inplace=True)

        # write chunk output
        if len(observations.index) > 0:
            PPWriteOutput(args, sconfigs, observations, verbose=False)
            if args.stats is not None:
                stats(observations, args.stats, args.outpath, sconfigs)
            has_output = True

        loopCounter += 1

    # Return the paths that this worker created (caller will merge them).
    if not has_output:
        return None, None

    # The actual file path from args so the caller can find them.
    ext = _output_extension(sconfigs.output.output_format)
    out_file = os.path.join(args.outpath, args.outfilestem + ext)
    stats_file = None
    if args.stats is not None:
        stats_file = os.path.join(args.outpath, args.outfilestem + "_stats" + ext)

    return out_file, stats_file


def _output_extension(output_format):
    """Map a Sorcha output format string to a file extension."""
    fmt = output_format.casefold()
    if fmt == "sqlite3":
        return ".db"
    if fmt in ("hdf5", "h5"):
        return ".h5"
    return ".csv"


def _merge_worker_outputs(worker_files, final_stem, outpath, output_format):
    """
    Concatenate per-worker output files into a single final file then delete
    the temporary worker files.

    Parameters
    ----------
    worker_files : list of str
        Paths to the temporary per-worker files (may contain None entries for
        workers that produced no observations — these are skipped).
    final_stem : str
        Output file stem for the merged file (no extension).
    outpath : str
    output_format : str
        Sorcha output format identifier.
    """
    valid = [f for f in worker_files if f is not None and os.path.isfile(f)]
    if not valid:
        return

    fmt = output_format.casefold()
    final_path = os.path.join(outpath, final_stem + _output_extension(output_format))

    if fmt == "sqlite3":
        import sqlite3

        # Copy all worker databases row-by-row into the final database.
        con_out = sqlite3.connect(final_path)
        for src_path in valid:
            con_src = sqlite3.connect(src_path)
            for line in con_src.iterdump():
                if line.startswith("INSERT"):
                    try:
                        con_out.execute(line)
                    except sqlite3.IntegrityError:
                        pass  # skip duplicate primary keys if any
            con_src.close()
        con_out.commit()
        con_out.close()

    elif fmt in ("hdf5", "h5"):
        frames = [pd.read_hdf(f) for f in valid]
        pd.concat(frames, ignore_index=True).to_hdf(final_path, key="df", mode="w")

    else:  # default: CSV
        frames = [pd.read_csv(f) for f in valid]
        pd.concat(frames, ignore_index=True).to_csv(final_path, index=False)

    # Clean up worker temp files.
    for f in valid:
        try:
            os.remove(f)
        except OSError:
            pass


def _fork_rngs(base_rngs, worker_id):
    """
    Derive a set of RNG objects for a worker process that are statistically
    independent from the base RNGs and from other workers.

    This uses numpy's SeedSequence spawning, which guarantees independence
    even when worker_id values are sequential integers.

    Parameters
    ----------
    base_rngs : object
        The RNG container from the top-level args.  Expected to expose the
        numpy RNG objects used by Sorcha.
    worker_id : int

    Returns
    -------
    forked_rngs : same type as base_rngs, with independent state
    """
    import copy

    forked = copy.deepcopy(base_rngs)

    # Walk every attribute that looks like a numpy Generator and reseed it.
    for attr_name in dir(forked):
        if attr_name.startswith("_"):
            continue
        val = getattr(forked, attr_name, None)
        if isinstance(val, np.random.Generator):
            # Derive a child seed that is a function of both the parent's
            # initial seed (if recoverable) and the worker_id.
            ss = np.random.SeedSequence(worker_id)
            setattr(forked, attr_name, np.random.default_rng(ss))

    return forked


def _run_worker_chunk_des(worker_id, obj_id_subset, args, sconfigs, filterpointing):
    """
    DES equivalent of _run_worker_chunk.  Runs a subset of objects through
    the full DES pipeline inside a worker process.

    Differences from the LSST worker:
      * No PPAddUncertainties call (removed for DES)
      * Uses DESFadingFunctionFilter instead of PPFadingFunctionFilter
      * Uses distance_cut / motion_cut / DESDiscoveryFilter instead of PPLinkingFilter

    Parameters
    ----------
    worker_id : int
    obj_id_subset : list
    args : sorchaArguments  (deepcopy with worker-specific outfilestem)
    sconfigs : sorchaConfigs
    filterpointing : pd.DataFrame

    Returns
    -------
    out_file : str or None
    stats_file : str or None
    """
    from sorcha.modules.PPDistanceandMotionCuts import distance_cut, motion_cut
    from sorcha.modules.DESDiscoveryFilter import DESDiscoveryFilter
    from sorcha.modules.DESFadingFunctionFilter import DESFadingFunctionFilter

    pplogger = logging.getLogger(__name__)

    footprint = None
    if sconfigs.fov.camera_model == "footprint":
        footprint = Footprint(sconfigs.fov.footprint_path, args.surveyname)

    reader = _build_reader(args, sconfigs)

    chunk_size = sconfigs.input.size_serial_chunk
    id_sub_chunks = [obj_id_subset[i : i + chunk_size] for i in range(0, len(obj_id_subset), chunk_size)]

    loopCounter = 0
    has_output = False

    for obj_id_chunk in id_sub_chunks:
        if sconfigs.input.ephemerides_type.casefold() == "external":
            observations = _read_chunk_by_ids(reader, obj_id_chunk, sconfigs)
        else:
            orbits_df = _read_chunk_by_ids(reader, obj_id_chunk, sconfigs)

            if not sconfigs.expert.brute_force:
                orbits_df = PPFaintObjectCullingFilter(
                    orbits_df,
                    filterpointing,
                    sconfigs.filters.mainfilter,
                    sconfigs.filters.observing_filters,
                    sconfigs.lightcurve.lc_model,
                    sconfigs.activity.comet_activity,
                )
                if len(orbits_df) == 0:
                    pplogger.info(
                        f"[worker {worker_id}] No objects pass faint-object culling for chunk "
                        f"{loopCounter}. Skipping."
                    )
                    loopCounter += 1
                    continue

            observations = create_ephemeris(orbits_df, filterpointing, args, sconfigs)

        if len(observations.index) == 0:
            pplogger.info(f"[worker {worker_id}] No ephemeris observations in chunk {loopCounter}. Skipping.")
            loopCounter += 1
            continue

        observations = PPMatchPointingToObservations(observations, filterpointing)

        observations = PPCalculateApparentMagnitude(
            observations,
            sconfigs.phasecurves.phase_function,
            sconfigs.filters.mainfilter,
            sconfigs.filters.othercolours,
            sconfigs.filters.observing_filters,
            sconfigs.activity.comet_activity,
            lightcurve_choice=sconfigs.lightcurve.lc_model,
            verbose=False,
        )

        if sconfigs.expert.trailing_losses_on:
            dmagDetect = PPTrailingLoss(observations, "circularPSF")
            observations["PSFMagTrue"] = dmagDetect + observations["trailedSourceMagTrue"]
        else:
            observations["PSFMagTrue"] = observations["trailedSourceMagTrue"]

        if sconfigs.expert.vignetting_on:
            observations["fiveSigmaDepth_mag"] = PPVignetting.vignettingEffects(observations)
        else:
            observations["fiveSigmaDepth_mag"] = observations["fieldFiveSigmaDepth_mag"]

        # PPAddUncertainties is intentionally omitted for DES

        if sconfigs.expert.randomization_on:
            observations = PPRandomizeMeasurements.randomizeAstrometryAndPhotometry(
                observations, sconfigs, args._rngs, verbose=False
            )
        else:
            observations["RATrue_deg"] = observations["RA_deg"].copy()
            observations["DecTrue_deg"] = observations["Dec_deg"].copy()
            observations["trailedSourceMag"] = observations["trailedSourceMagTrue"].copy()
            observations["PSFMag"] = observations["PSFMagTrue"].copy()

        if sconfigs.fov.camera_model != "none" and len(observations.index) > 0:
            observations = PPApplyFOVFilter(
                observations,
                sconfigs,
                args._rngs,
                visits=args.visits,
                footprint=footprint,
                verbose=False,
            )

        if sconfigs.expert.snr_limit_on and len(observations.index) > 0:
            observations = PPSNRLimit(observations, sconfigs.expert.snr_limit)

        if sconfigs.expert.mag_limit_on and len(observations.index) > 0:
            observations = PPMagnitudeLimit(observations, sconfigs.expert.mag_limit)

        if sconfigs.fadingfunction.fading_function_on and len(observations.index) > 0:
            observations = DESFadingFunctionFilter(
                observations,
                sconfigs.fadingfunction.des_transient_efficency,
                args._rngs,
                verbose=False,
            )

        if sconfigs.saturation.bright_limit_on and len(observations.index) > 0:
            observations = PPBrightLimit(
                observations, sconfigs.filters.observing_filters, sconfigs.saturation.bright_limit
            )

        if sconfigs.linkingfilter.distance_cut_on and len(observations.index) > 0:
            observations = distance_cut(
                observations,
                sconfigs.linkingfilter.distance_cut_upper,
                sconfigs.linkingfilter.distance_cut_lower,
            )

        if sconfigs.linkingfilter.motion_cut_on and len(observations.index) > 0:
            observations = motion_cut(
                observations,
                sconfigs.linkingfilter.motion_cut_upper,
                sconfigs.linkingfilter.motion_cut_lower,
            )

        if sconfigs.linkingfilter.des_discovery_on and len(observations.index) > 0:
            observations = DESDiscoveryFilter(observations)

        if len(observations.index) > 0:
            PPWriteOutput(args, sconfigs, observations, verbose=False)
            if args.stats is not None:
                stats(observations, args.stats, args.outpath, sconfigs)
            has_output = True

        loopCounter += 1

    if not has_output:
        return None, None

    ext = _output_extension(sconfigs.output.output_format)
    out_file = os.path.join(args.outpath, args.outfilestem + ext)
    stats_file = None
    if args.stats is not None:
        stats_file = os.path.join(args.outpath, args.outfilestem + "_stats" + ext)

    return out_file, stats_file
