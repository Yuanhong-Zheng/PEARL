import os
from datasets import load_dataset
from tqdm.auto import tqdm

# (repo, subset, split)
EVAL_DATASET_HF_PATH = {
    # Video-RET
    "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
    "MSVD": ("VLM2Vec/MSVD", None, "test"),
    "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
    # "YouCook2": ("VLM2Vec/YouCook2", None, "val"), # HF version compatibility issue
    "YouCook2": ("lmms-lab/YouCook2", None, "val"),
    "VATEX": ("VLM2Vec/VATEX", None, "test"),

    # Video-CLS
    "HMDB51": ("VLM2Vec/HMDB51", None, "test"),
    "UCF101": ("VLM2Vec/UCF101", None, "test"),
    "Breakfast": ("VLM2Vec/Breakfast", None, "test"),
    "Kinetics-700": ("VLM2Vec/Kinetics-700", None, "test"),
    "SmthSmthV2": ("VLM2Vec/SmthSmthV2", None, "test"),

    # Video-MRET
    "QVHighlight": ("VLM2Vec/QVHighlight", None, "test"),
    "Charades-STA": ("VLM2Vec/Charades-STA", None, "test"),
    "MomentSeeker": ("VLM2Vec/MomentSeeker", None, "test"),
    "MomentSeeker_1k8": ("VLM2Vec/MomentSeeker_1k8", None, "test"),

    # Video-QA
    "NExTQA": ("VLM2Vec/NExTQA", "MC", "test"),
    "EgoSchema": ("VLM2Vec/EgoSchema", "Subset", "test"),
    "MVBench": ("VLM2Vec/MVBench", None, "train"),
    "Video-MME": ("VLM2Vec/Video-MME", None, "test"),
    "ActivityNetQA": ("VLM2Vec/ActivityNetQA", None, "test"),

    # Visdoc-ViDoRe
    "ViDoRe_arxivqa": ("vidore/arxivqa_test_subsampled_beir", None, "test"),
    "ViDoRe_docvqa": ("vidore/docvqa_test_subsampled_beir", None, "test"),
    "ViDoRe_infovqa": ("vidore/infovqa_test_subsampled_beir", None, "test"),
    "ViDoRe_tabfquad": ("vidore/tabfquad_test_subsampled_beir", None, "test"),
    "ViDoRe_tatdqa": ("vidore/tatdqa_test_beir", None, "test"),
    "ViDoRe_shiftproject": ("vidore/shiftproject_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_artificial_intelligence": ("vidore/syntheticDocQA_artificial_intelligence_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_energy": ("vidore/syntheticDocQA_energy_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_government_reports": ("vidore/syntheticDocQA_government_reports_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_healthcare_industry": ("vidore/syntheticDocQA_healthcare_industry_test_beir", None, "test"),

    # Visdoc-VisRAG
    "VisRAG_ArxivQA": ("openbmb/VisRAG-Ret-Test-ArxivQA", None, "train"),
    "VisRAG_ChartQA": ("openbmb/VisRAG-Ret-Test-ChartQA", None, "train"),
    "VisRAG_MP-DocVQA": ("openbmb/VisRAG-Ret-Test-MP-DocVQA", None, "train"),
    "VisRAG_SlideVQA": ("openbmb/VisRAG-Ret-Test-SlideVQA", None, "train"),
    "VisRAG_InfoVQA": ("openbmb/VisRAG-Ret-Test-InfoVQA", None, "train"),
    "VisRAG_PlotQA": ("openbmb/VisRAG-Ret-Test-PlotQA", None, "train"),

    # Visdoc-ViDoSeek
    "ViDoSeek-doc": ("VLM2Vec/ViDoSeek", None, "test"),
    "ViDoSeek-page": ("VLM2Vec/ViDoSeek-page-fixed", None, "test"),
    "MMLongBench-doc": ("VLM2Vec/MMLongBench-doc", None, "test"),
    "MMLongBench-page": ("VLM2Vec/MMLongBench-page-fixed", None, "test"),

    # Visdoc-ViDoRe_v2
    "ViDoRe_esg_reports_human_labeled_v2": ("vidore/esg_reports_human_labeled_v2", None, "test"),
    "ViDoRe_biomedical_lectures_v2": ("vidore/biomedical_lectures_v2", "english", "test"),
    "ViDoRe_biomedical_lectures_v2_multilingual": ("vidore/biomedical_lectures_v2", None, "test"),
    "ViDoRe_economics_reports_v2": ("vidore/economics_reports_v2", "english", "test"),
    "ViDoRe_economics_reports_v2_multilingual": ("vidore/economics_reports_v2", None, "test"),
    "ViDoRe_esg_reports_v2": ("vidore/esg_reports_v2", "english", "test"),
    "ViDoRe_esg_reports_v2_multilingual": ("vidore/esg_reports_v2", None, "test"),

    # "ViDoRe_esg_reports_v2":("vidore/synthetic_rse_restaurant_filtered_v1.0", None, "test"),
    # "ViDoRe_esg_reports_v2_multilingual":("vidore/synthetic_rse_restaurant_filtered_v1.0_multilingual", None, "test"),
    # "ViDoRe_biomedical_lectures_v2":("vidore/synthetic_mit_biomedical_tissue_interactions_unfiltered", None, "test"),
    # "ViDoRe_biomedical_lectures_v2_multilingual":("vidore/synthetic_mit_biomedical_tissue_interactions_unfiltered_multilingual", None, "test"),
    # "ViDoRe_economics_reports_v2":("vidore/synthetic_economics_macro_economy_2024_filtered_v1.0", None, "test"),
    # "ViDoRe_economics_reports_v2_multilingual":("vidore/synthetic_economics_macro_economy_2024_filtered_v1.0_multilingual", None, "test"),
    # "ViDoRe_esg_reports_human_labeled_v2":("vidore/esg_reports_human_labeled_v2", None, "test"),
}

SUBSET_MAP = {
    'MVBench': {
        'object_interaction', 'moving_count', 'moving_attribute', 
        'scene_transition', 'object_existence', 'episodic_reasoning', 
        'action_antonym', 'character_order', 'action_localization', 
        'action_prediction', 'moving_direction', 'action_count', 
        'state_change', 'fine_grained_pose', 'unexpected_action', 
        'object_shuffle', 'counterfactual_inference', 'action_sequence', 
        'fine_grained_action', 'egocentric_navigation'
    },
    'ViDoRe_arxivqa': {'corpus', 'qrels', 'queries'},
    'ViDoRe_docvqa': {'corpus', 'qrels', 'queries'},
    'ViDoRe_infovqa': {'corpus', 'qrels', 'queries'},
    'ViDoRe_tabfquad': {'corpus', 'qrels', 'queries'},
    'ViDoRe_tatdqa': {'corpus', 'qrels', 'queries'},
    'ViDoRe_shiftproject': {'corpus', 'qrels', 'queries'},
    'ViDoRe_syntheticDocQA_artificial_intelligence': {'corpus', 'qrels', 'queries'},
    'ViDoRe_syntheticDocQA_energy': {'corpus', 'qrels', 'queries'},
    'ViDoRe_syntheticDocQA_government_reports': {'corpus', 'qrels', 'queries'},
    'ViDoRe_syntheticDocQA_healthcare_industry': {'corpus', 'qrels', 'queries'},
    'VisRAG_ArxivQA': {'corpus', 'qrels', 'queries'},
    'VisRAG_ChartQA': {'corpus', 'qrels', 'queries'},
    'VisRAG_MP-DocVQA': {'corpus', 'qrels', 'queries'},
    'VisRAG_SlideVQA': {'corpus', 'qrels', 'queries'},
    'VisRAG_InfoVQA': {'corpus', 'qrels', 'queries'},
    'VisRAG_PlotQA': {'corpus', 'qrels', 'queries'},
    'ViDoSeek-doc': {'corpus', 'qrels', 'queries'},
    'ViDoSeek-page': {'corpus', 'qrels', 'queries'},
    'MMLongBench-doc': {'corpus', 'qrels', 'queries'},
    'MMLongBench-page': {'corpus', 'qrels', 'queries'},
    'ViDoRe_esg_reports_human_labeled_v2': {'corpus', 'qrels', 'queries'},
    'ViDoRe_biomedical_lectures_v2': {'corpus', 'qrels', 'queries'},
    'ViDoRe_biomedical_lectures_v2_multilingual': {'corpus', 'qrels', 'queries'},
    'ViDoRe_economics_reports_v2': {'corpus', 'qrels', 'queries'},
    'ViDoRe_economics_reports_v2_multilingual': {'corpus', 'qrels', 'queries'},
    'ViDoRe_esg_reports_v2': {'corpus', 'qrels', 'queries'},
    'ViDoRe_esg_reports_v2_multilingual': {'corpus', 'qrels', 'queries'},
}

BASE_ANNOTATION_DIR = "data/evaluation/mmeb_v2/annotation"

EVAL_DATASET_LOCAL_PATH = {
    key: (
        os.path.join(BASE_ANNOTATION_DIR, val[0]),
        val[1], 
        val[2]
    )
    for key, val in EVAL_DATASET_HF_PATH.items()
}

def download_dataset(name, repo_id, subset, split):
    """Download specified subset and split of a single dataset"""
    local_path = os.path.join(BASE_ANNOTATION_DIR, repo_id)
    
    # Check if already exists
    if os.path.exists(local_path):
        print(f"⏭️  Skipping {name} - already exists at {local_path}")
        return
    
    print(f"📥 Downloading {name}: {repo_id} (subset={subset}, split={split})")

    try:
        # Load dataset with specified subset and split
        if name in SUBSET_MAP:
            for subset in SUBSET_MAP[name]:
                dataset = load_dataset(repo_id, subset, split=split) # , trust_remote_code=True
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                dataset.save_to_disk(os.path.join(local_path, subset))
        else:
            if subset:
                dataset = load_dataset(repo_id, subset, split=split) # , trust_remote_code=True
            else:
                dataset = load_dataset(repo_id, split=split) # , trust_remote_code=True
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            dataset.save_to_disk(local_path)

        print(f"✅ Successfully downloaded {name} to {local_path}")
        
    except Exception as e:
        print(f"❌ Error downloading {name}: {str(e)}")

def main():
    os.makedirs(BASE_ANNOTATION_DIR, exist_ok=True)
    
    print(f"Starting dataset downloads to {BASE_ANNOTATION_DIR}...")
    print(f"Total datasets: {len(EVAL_DATASET_HF_PATH)}\n")
    
    for name, (repo_id, subset, split) in tqdm(EVAL_DATASET_HF_PATH.items()):
        print(f"\n{'='*80}")
        download_dataset(name, repo_id, subset, split)
    
    print(f"\n{'='*80}")
    print("All downloads completed!")

if __name__ == "__main__":
    main()