# git clone https://github.com/forithmus/VLM3D-Dockers.git
# huggingface-cli download --resume-download zzxslp/RadBERT-RoBERTa-4m --local-dir /hdd/common/datasets/medical-image-analysis/CT-RATE/models/radbert_local
# reportgen_evaluation/dataset.py (L25): MODEL_DIR = Path("/hdd/common/datasets/medical-image-analysis/CT-RATE/models/radbert_local").resolve()
# reportgen_evaluation/classifier.py (L9): MODEL_DIR = Path("/hdd/common/datasets/medical-image-analysis/CT-RATE/models/radbert_local").resolve()
# reportgen_evaluation/calc_scores.py (L28): pred['AccessionNo'] = pred['AccessionNo'].str.replace('.nii',  '', regex=False)
# reportgen_evaluation/crg_score.py (L34): .str.replace('.nii', '', regex=False)
# reportgen_evaluation/nlg_metrics.py (L63): gt_items = json.load(f)[0]["outputs"][0]["value"]["generated_reports"]

CUDA_ID=3
EVAL_DIR=/home/gaoj/share4/vlm-finetune/trl/eval_ctrate_rg/reportgen_evaluation/ctrate-qwenvl-32-364-rg-my-batch-1.0-0.1-post-training-2-epoch

# echo "Formatting output..."
# python /home/shiym/projects/vlm-finetune/swift/scripts/format_output.py \
#     --model_path $EVAL_DIR

echo "[1/4] extracting cls results..."
CUDA_VISIBLE_DEVICES=$CUDA_ID python reportgen_evaluation/infer.py \
    --input_json $EVAL_DIR/output_format.json \
    --model_path /hdd/common/datasets/medical-image-analysis/CT-RATE/models/RadBertClassifier.pth \
    --out_csv $EVAL_DIR/output_cls.csv

echo "[2/4] calculating cls metrics..."
python reportgen_evaluation/calc_scores.py \
    --pred_csv $EVAL_DIR/output_cls.csv \
    --gt_csv reportgen_evaluation/ground-truth/ground_truth.csv \
    --out_json $EVAL_DIR/metrics_cls.json

echo "[3/4] calculating crg metrics..."
python reportgen_evaluation/crg_score.py \
    --pred_csv $EVAL_DIR/output_cls.csv \
    --gt_csv reportgen_evaluation/ground-truth/ground_truth.csv \
    --out_json $EVAL_DIR/metrics_crg.json

echo "[4/4] calculating nlg metrics..."
python reportgen_evaluation/nlg_metrics.py \
    --pred_json $EVAL_DIR/output_format.json \
    --gt_json reportgen_evaluation/ground-truth/ground_truth.json \
    --out_json $EVAL_DIR/metrics_nlg.json
