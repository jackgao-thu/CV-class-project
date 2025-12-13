import json
import os

def process_report(content):
    """处理报告内容，移除'Findings: '和'Impression:  '前缀"""
    # 移除开头的'Findings: '
    if content.startswith('Findings: '):
        content = content[len('Findings: '):]
    # 移除剩余的'Impression:  '（注意冒号后有两个空格）
    content = content.replace('Impression:  ', '')
    return content

def process_video_path(video_path):
    """处理视频路径，提取文件名并替换扩展名为.nii.gz"""
    # 获取文件名（包含扩展名）
    filename = os.path.basename(video_path)
    # 替换扩展名为.nii.gz
    return os.path.splitext(filename)[0] + '.nii.gz'

def convert_json(input_file, output_file):
    """转换JSON文件格式"""
    # 读取原始JSON数据
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 处理生成报告列表
    generated_reports = []
    for item in original_data:
        # 获取assistant的content
        assistant_content = next(
            msg['content'] for msg in item['messages'] 
            if msg['role'] == 'assistant'
        )
        # 处理报告内容
        processed_report = process_report(assistant_content)
        
        # 处理视频路径（假设每个item只有一个视频）
        video_path = item['videos'][0]
        image_name = process_video_path(video_path)
        
        # 添加到报告列表
        generated_reports.append({
            "input_image_name": image_name,
            "report": processed_report
        })
    
    # 构建目标格式数据
    target_data = [{
        "outputs": [{
            "value": {
                "generated_reports": generated_reports
            }
        }]
    }]
    
    # 写入目标JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 转换文件
    
    convert_json('/home/gaoj/share4/vlm-finetune/trl/ctrate-qwenvl-32-364-rg-my-batch-1.0-0.1-post-training-2-epoch/eval/output.json', '/home/gaoj/share4/vlm-finetune/trl/eval_ctrate_rg/reportgen_evaluation/ctrate-qwenvl-32-364-rg-my-batch-1.0-0.1-post-training-2-epoch/output_format.json')
    print("转换完成，结果已保存到output_format.json")