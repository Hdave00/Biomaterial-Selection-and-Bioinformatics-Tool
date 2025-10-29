

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
import sys

# If utils is not on sys.path when run from project root, ensure it is.
# (This helps when you call DataProcessor from app code.)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .master_index import create_material_index  # your existing function

BASE_MASTER_DIR = Path("master_data")
BASE_MASTER_DIR.mkdir(exist_ok=True)


class DataProcessor:
    """
    Flexible orchestrator that runs/loads pipeline outputs and returns unified ML-ready data.
    It will attempt to call common entry-points in modules to obtain DataFrames.
    """

    # For each domain we keep a list of candidate function names in order of preference
    PIPELINE_FN_CANDIDATES = {
        "biological": [
            "run_biological_pipeline", "create_cytotoxicity_scores",
            "create_cytotoxicity_score", "load_biological_data", "load_data"
        ],
        "chemical": [
            "run_chemical_pipeline", "create_chemical_raw", "create_chemical_features",
            "create_corrosion_scores", "load_chemical_data", "load_data"
        ],
        "corrosion": [
            "run_corrosion_pipeline", "create_corrosion_compatibility",
            "create_corrosion_scores", "load_data"
        ],
        "polymer": [
            # For polymer we know the file exposes load_data / merge_polymer_data / clean_for_ml
            # but keep fallback names too.
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

    # -------------------------
    # Helper: dynamic function runner
    # -------------------------
    @staticmethod
    def _call_first_available(module, candidates, *args, **kwargs):
        """
        Given a module and a list of candidate function names (strings),
        call the first function that exists in the module.
        Return the result of the call. If nothing found, return None.
        """
        for name in candidates:
            if hasattr(module, name):
                fn = getattr(module, name)
                if callable(fn):
                    # If function expects no args -> call without
                    try:
                        sig = inspect.signature(fn)
                        # Support functions that accept nothing or accept DataFrames/paths,
                        # attempt a safe call with args/kwargs when needed.
                        if len(sig.parameters) == 0:
                            return fn()
                        else:
                            # If caller passed args/kwargs, try calling with them
                            return fn(*args, **kwargs)
                    except Exception as e:
                        # try calling without args as a fallback
                        try:
                            return fn()
                        except Exception as e2:
                            # both attempts fail, raise so the user sees the problem
                            raise
        return None

    # -------------------------
    # Domain loaders (flexible)
    # -------------------------
    def _load_biological(self):
        try:
            mod = importlib.import_module("utils.pipelines.biological_pipeline")
            df = self._call_first_available(mod, self.PIPELINE_FN_CANDIDATES["biological"])
            if isinstance(df, pd.DataFrame):
                self.biological = df
                return df
        except ModuleNotFoundError:
            # fallback to the older single-file merge that you might have
            try:
                mod = importlib.import_module("utils.merge_biological_data")
                df = self._call_first_available(mod, self.PIPELINE_FN_CANDIDATES["biological"])
                if isinstance(df, pd.DataFrame):
                    self.biological = df
                    return df
            except ModuleNotFoundError:
                pass
        return None

    def _load_chemical(self):
        try:
            mod = importlib.import_module("utils.pipelines.chemical_pipeline")
            df = self._call_first_available(mod, self.PIPELINE_FN_CANDIDATES["chemical"])
            if isinstance(df, pd.DataFrame):
                self.chemical = df
                return df
        except ModuleNotFoundError:
            try:
                mod = importlib.import_module("utils.merge_chemical_data")
                df = self._call_first_available(mod, self.PIPELINE_FN_CANDIDATES["chemical"])
                if isinstance(df, pd.DataFrame):
                    self.chemical = df
                    return df
            except ModuleNotFoundError:
                pass
        return None

    def _load_corrosion(self):
        try:
            mod = importlib.import_module("utils.pipelines.corrosion_pipeline")
            df = self._call_first_available(mod, self.PIPELINE_FN_CANDIDATES["corrosion"])
            if isinstance(df, pd.DataFrame):
                self.corrosion = df
                return df
        except ModuleNotFoundError:
            try:
                mod = importlib.import_module("utils.merge_chemical_data")
                df = self._call_first_available(mod, self.PIPELINE_FN_CANDIDATES["corrosion"])
                if isinstance(df, pd.DataFrame):
                    self.corrosion = df
                    return df
            except ModuleNotFoundError:
                pass
        return None

    def _load_polymer(self):

        # handle merge_polymer_ml_data.py specifically (it exposes load_data, merge, clean)
        try:
            mod = importlib.import_module("utils.merge_polymer_ml_data")
            # prefer explicit step-by-step call using load_data() then merge then clean_for_ml()
            if hasattr(mod, "load_data") and hasattr(mod, "merge_polymer_data"):
                df_tg, df_main = mod.load_data()
                merged = mod.merge_polymer_data(df_tg, df_main)
                if hasattr(mod, "clean_for_ml"):
                    merged = mod.clean_for_ml(merged)
                self.polymer = merged
                return merged
            # fallback: call a single create_* function
            df = self._call_first_available(mod, self.PIPELINE_FN_CANDIDATES["polymer"])
            if isinstance(df, pd.DataFrame):
                self.polymer = df
                return df
        except ModuleNotFoundError:
            # try pipelines directory
            try:
                mod = importlib.import_module("utils.pipelines.polymer_pipeline")
                df = self._call_first_available(mod, self.PIPELINE_FN_CANDIDATES["polymer"])
                if isinstance(df, pd.DataFrame):
                    self.polymer = df
                    return df
            except ModuleNotFoundError:
                pass
        return None

    def _load_mechanical(self):
        try:
            mod = importlib.import_module("utils.merge_material_csv_data")
            # this file returns a unified DF if you run merge script; see if it exposes a function
            df = self._call_first_available(mod, self.PIPELINE_FN_CANDIDATES["mechanical"])
            if df is None:
                # as fallback attempt to load the output CSV produced by the script
                out = Path("data/materials_data/unified_material_data.csv")
                if out.exists():
                    df = pd.read_csv(out)
            if isinstance(df, pd.DataFrame):
                self.mechanical = df
                return df
        except ModuleNotFoundError:
            pass
        return None

    # -------------------------
    # Master index and linking
    # -------------------------
    def build_master_index(self):
        """
        Build or load master index. We try to call your create_material_index()
        which (per earlier version) may be self-contained.
        """
        try:
            idx = create_material_index()
            if isinstance(idx, pd.DataFrame) and "material_name" in idx.columns:
                self.master_index = idx
                return idx
        except Exception:
            # Last-resort: create a simple index from available data
            names = set()
            for df in [self.mechanical, self.chemical, self.corrosion, self.polymer, self.biological]:
                if isinstance(df, pd.DataFrame):
                    # try common name columns
                    for c in ["Material", "material_name", "Polymer", "name", "Material_Raw", "Material_Name"]:
                        if c in df.columns:
                            names.update(df[c].dropna().astype(str).str.upper().unique())
            master_index = pd.DataFrame(sorted(list(names)), columns=["material_name"])
            master_index["material_id"] = master_index["material_name"].apply(
                lambda x: hashlib.md5(x.encode()).hexdigest().upper()
            )
            self.master_index = master_index
            return master_index

    # -------------------------
    # Public orchestration method
    # -------------------------
    def process_all_data(self):
        print("DataProcessor: loading mechanical data...")
        self._load_mechanical()

        print("DataProcessor: loading polymer data...")
        self._load_polymer()

        print("DataProcessor: loading chemical data...")
        self._load_chemical()

        print("DataProcessor: loading corrosion data...")
        self._load_corrosion()

        print("DataProcessor: loading biological data...")
        self._load_biological()

        print("DataProcessor: building master index...")
        self.build_master_index()

        # Link datasets by normalized material name -> master_index.material_name
        self._link_datasets()

        # Create a final unified file (simple concat of domain outputs keyed by material_name)
        self._create_unified_master_file()

        return {
            "mechanical": self.mechanical,
            "polymer": self.polymer,
            "chemical": self.chemical,
            "corrosion": self.corrosion,
            "biological": self.biological,
            "master_index": self.master_index,
            "unified": self.unified
        }

    def _link_datasets(self):
        """Add material_name / material_id fields to each dataset where possible."""
        if self.master_index is None:
            return

        def try_map(df):
            if df is None or not isinstance(df, pd.DataFrame):
                return df
            df = df.copy()
            # find best name column
            for c in ["material_name", "Material", "Material_Raw", "Material_Name", "name", "Polymer", "Polymer_Class"]:
                if c in df.columns:
                    df["material_name"] = df[c].astype(str).str.upper()
                    break
            # left-join with master index to get material_id
            df = df.merge(self.master_index, on="material_name", how="left")
            return df

        self.mechanical = try_map(self.mechanical)
        self.polymer = try_map(self.polymer)
        self.chemical = try_map(self.chemical)
        self.corrosion = try_map(self.corrosion)
        self.biological = try_map(self.biological)

    def _create_unified_master_file(self):
        """Combine cleaned domain outputs into a single CSV for ML consumption."""
        pieces = []
        for df in [self.mechanical, self.polymer, self.chemical, self.corrosion, self.biological]:
            if isinstance(df, pd.DataFrame):
                pieces.append(df)
        if not pieces:
            self.unified = None
            return

        # concat and drop exact duplicates
        unified = pd.concat(pieces, ignore_index=True, sort=False)
        # ensure canonical columns exist
        if "material_name" in unified.columns and "material_id" in unified.columns:
            # drop rows lacking an id
            unified = unified.dropna(subset=["material_id"])
            unified = unified.drop_duplicates(subset=["material_id", "material_name"])
        unified.to_csv(BASE_MASTER_DIR / "unified_material_data.csv", index=False)
        self.unified = unified
        print(f"Saved unified master CSV -> {BASE_MASTER_DIR / 'unified_material_data.csv'}")
