"""
From a script or interactive session:

from utils.data_processor import DataProcessor

dp = DataProcessor()
res = dp.process_all_data()
# res is a dict containing domain DataFrames + master index + unified table
"""

# utils/data_processor.py
import importlib
import inspect
from pathlib import Path
import pandas as pd
import hashlib
import logging
import sys
from datetime import datetime


# Setup: project paths & logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_MASTER_DIR = ROOT / "master_data"
BASE_MASTER_DIR.mkdir(exist_ok=True)

# Configure logging
LOG_FILE = BASE_MASTER_DIR / f"data_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# Import master index creator
try:
    from .master_index import create_material_index
except ImportError:
    create_material_index = None
    logger.warning("Could not import create_material_index. Will fall back to basic index builder.")


class DataProcessor:

    """
    Master orchestrator that dynamically runs domain pipelines (biological, chemical, etc.)
    to create a unified ML usable dataset and master index.
    """

    PIPELINE_FN_CANDIDATES = {
        "biological": [
            "run_biological_pipeline", "create_cytotoxicity_scores",
            "create_cytotoxicity_score", "load_biological_data", "load_data"
        ],
        "chemical": [
            "run_chemical_pipeline", "create_chemical_raw",
            "create_chemical_features", "create_corrosion_scores",
            "load_chemical_data", "load_data"
        ],
        "corrosion": [
            "run_corrosion_pipeline", "create_corrosion_compatibility",
            "create_corrosion_scores", "load_data"
        ],
        "polymer": [
            "create_unified_polymer_ml_data", "merge_polymer_data",
            "create_unified_polymer_data", "load_data"
        ],
        "mechanical": [
            "create_mechanical_unified", "run_materials_pipeline", "load_data"
        ]
    }

    def __init__(self):
        self.biological = None
        self.chemical = None
        self.corrosion = None
        self.polymer = None
        self.mechanical = None
        self.master_index = None
        self.unified = None


    # Helper: dynamic function runner
    @staticmethod
    def _call_first_available(module, candidates, *args, **kwargs):
        for name in candidates:
            if hasattr(module, name):
                fn = getattr(module, name)
                if callable(fn):
                    try:
                        sig = inspect.signature(fn)
                        logger.info(f"🔹 Calling {module.__name__}.{name}()")
                        if len(sig.parameters) == 0:
                            return fn()
                        else:
                            return fn(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in {module.__name__}.{name}: {e}", exc_info=True)
        logger.warning(f"No valid function found among {candidates} in {module.__name__}")
        return None


    # Domain loaders
    def _load_domain(self, domain, module_paths):

        """
        Generic loader for a given domain.
        Tries each module path until a DataFrame is returned.
        """
        candidates = self.PIPELINE_FN_CANDIDATES.get(domain, [])

        for mod_path in module_paths:
            try:

                mod = importlib.import_module(mod_path)
                logger.info(f"Loaded module {mod_path}")
                df = self._call_first_available(mod, candidates)

                if isinstance(df, pd.DataFrame):

                    logger.info(f"{domain.capitalize()} data loaded successfully ({len(df)} rows)")
                    setattr(self, domain, df)

                    return df
                
            except ModuleNotFoundError:
                logger.warning(f"Module not found: {mod_path}")
            except Exception as e:
                logger.error(f"Failed to load {domain} from {mod_path}: {e}", exc_info=True)

        logger.warning(f"No valid data found for {domain}")
        return None

    def _load_mechanical(self):
        return self._load_domain("mechanical", [
            "utils.pipelines.mechanical_pipeline",
            "utils.merge_material_csv_data"
        ])

    def _load_polymer(self):

        try:

            mod = importlib.import_module("utils.merge_polymer_ml_data")

            if hasattr(mod, "load_data") and hasattr(mod, "merge_polymer_data"):

                df_tg, df_main = mod.load_data()
                merged = mod.merge_polymer_data(df_tg, df_main)

                if hasattr(mod, "clean_for_ml"):
                    merged = mod.clean_for_ml(merged)

                self.polymer = merged
                logger.info(f"Polymer data merged successfully ({len(merged)} rows)")

                return merged
            
        except Exception as e:
            logger.warning(f"Fallback to pipeline for polymer: {e}")
        return self._load_domain("polymer", [
            "utils.pipelines.polymer_pipeline",
            "utils.merge_polymer_ml_data"
        ])

    def _load_chemical(self):
        return self._load_domain("chemical", [
            "utils.pipelines.chemical_pipeline",
            "utils.merge_chemical_data"
        ])

    def _load_corrosion(self):
        return self._load_domain("corrosion", [
            "utils.pipelines.corrosion_pipeline",
            "utils.merge_corrosion_data"
        ])

    def _load_biological(self):
        return self._load_domain("biological", [
            "utils.pipelines.biological_pipeline",
            "utils.merge_biological_data"
        ])


    # Master index builder
    def build_master_index(self):

        try:

            if create_material_index:
                idx = create_material_index()

                if isinstance(idx, pd.DataFrame):
                    self.master_index = idx
                    logger.info(f"Master index built using create_material_index ({len(idx)} entries)")
                    return idx
                
        except Exception as e:
            logger.error(f"create_material_index failed: {e}", exc_info=True)

        # Fallback: construct from all known data
        names = set()

        for df in [self.mechanical, self.chemical, self.corrosion, self.polymer, self.biological]:
            if isinstance(df, pd.DataFrame):

                for c in ["Material", "material_name", "Polymer", "name", "Material_Raw", "Material_Name"]:
                    if c in df.columns:
                        names.update(df[c].dropna().astype(str).str.upper().unique())

        master_index = pd.DataFrame(sorted(list(names)), columns=["material_name"])
        master_index["material_id"] = master_index["material_name"].apply(
            lambda x: hashlib.md5(x.encode()).hexdigest().upper()
)

        self.master_index = master_index
        logger.info(f"Master index built from dataset names ({len(master_index)} entries)")
        return master_index


    # Linking datasets
    def _link_datasets(self):

        if self.master_index is None:
            return

        def try_map(df):
            if df is None or not isinstance(df, pd.DataFrame):
                return df
            df = df.copy()
            for c in ["material_name", "Material", "Material_Raw", "Material_Name", "name", "Polymer", "Polymer_Class"]:
                if c in df.columns:
                    df["material_name"] = df[c].astype(str).str.upper()
                    break
            df = df.merge(self.master_index, on="material_name", how="left")
            return df

        self.mechanical = try_map(self.mechanical)
        self.polymer = try_map(self.polymer)
        self.chemical = try_map(self.chemical)
        self.corrosion = try_map(self.corrosion)
        self.biological = try_map(self.biological)
        logger.info("🔗 Linked all domain datasets to master index")


    # Final unified file
    def _create_unified_master_file(self):

        pieces = [df for df in [self.mechanical, self.polymer, self.chemical, self.corrosion, self.biological]
                  if isinstance(df, pd.DataFrame)]
        
        if not pieces:
            logger.warning("No datasets to unify.")
            self.unified = None
            return

        unified = pd.concat(pieces, ignore_index=True, sort=False)

        if "material_name" in unified.columns and "material_id" in unified.columns:
            unified = unified.dropna(subset=["material_id"])
            unified = unified.drop_duplicates(subset=["material_id", "material_name"])

        output_path = BASE_MASTER_DIR / "unified_material_data.csv"
        unified.to_csv(output_path, index=False)
        self.unified = unified
        logger.info(f"💾 Saved unified master CSV -> {output_path} ({len(unified)} rows)")


    # Public runner
    def process_all_data(self):

        logger.info("Starting full data processing pipeline")

        self._load_mechanical()
        self._load_polymer()
        self._load_chemical()
        self._load_corrosion()
        self._load_biological()
        self.build_master_index()
        self._link_datasets()
        self._create_unified_master_file()
        logger.info(" Data processing complete!")

        return {
            "mechanical": self.mechanical,
            "polymer": self.polymer,
            "chemical": self.chemical,
            "corrosion": self.corrosion,
            "biological": self.biological,
            "master_index": self.master_index,
            "unified": self.unified
        }
