from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from model import load_trained_model, predict_single_pressure_map


app = Flask(__name__)
CORS(app)  # 解决跨域问题

# 模型输出数据根目录
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "model_output")
# 前端静态文件目录（图片要存在这里，前端才能访问）
FRONTEND_STATIC_DIR = os.path.join(os.path.dirname(__file__), "../frontend/static")
# 图片在前端静态目录下的子目录（按人员分类）
HEATMAP_IMAGE_DIR = os.path.join(FRONTEND_STATIC_DIR, "heatmap_images")
os.makedirs(HEATMAP_IMAGE_DIR, exist_ok=True)  # 确保目录存在


try:
    model = load_trained_model("backend/models/svm_model.pkl")
    print("模型加载成功，可进行预测")
except Exception as e:
    print(f"模型加载失败：{e}")
    model = None

# 接口1：获取所有人员列表（不变）
@app.route("/api/people", methods=["GET"])
def get_people():
    try:
        people = [
            d
            for d in os.listdir(MODEL_OUTPUT_DIR)
            if os.path.isdir(os.path.join(MODEL_OUTPUT_DIR, d))
        ]
        return jsonify({"code": 200, "data": people, "msg": "获取人员列表成功"})
    except Exception as e:
        return jsonify({"code": 500, "msg": f"获取人员列表失败：{str(e)}"})


# 接口2：获取指定人员、指定姿势的热力图数据（不变）
@app.route("/api/heatmap", methods=["GET"])
def get_heatmap():
    person = request.args.get("person")
    posture_id = request.args.get("posture_id")
    if not person or not posture_id:
        return jsonify({"code": 400, "msg": "缺少person或posture_id参数"})

    try:
        file_path = os.path.join(MODEL_OUTPUT_DIR, person, f"posture_{posture_id}.npy")
        if not os.path.exists(file_path):
            return jsonify({"code": 404, "msg": f"该人员无此姿势数据"})

        pressure_matrix = np.load(file_path).astype(np.float32).tolist()
        return jsonify(
            {
                "code": 200,
                "data": {"matrix": pressure_matrix, "rows": 40, "cols": 26},
                "msg": "获取热力图数据成功",
            }
        )
    except Exception as e:
        return jsonify({"code": 500, "msg": f"获取数据失败：{str(e)}"})


# 新增接口3：生成图片并保存到本地（供前端触发生成）
@app.route("/api/generate_heatmap_image", methods=["GET"])
def generate_heatmap_image():
    person = request.args.get("person")
    posture_id = request.args.get("posture_id")
    if not person or not posture_id:
        return jsonify({"code": 400, "msg": "缺少person或posture_id参数"})

    try:
        # 1. 加载压力数据（省略，和之前逻辑一致）
        file_path = os.path.join(MODEL_OUTPUT_DIR, person, f"posture_{posture_id}.npy")
        pressure_matrix = np.load(file_path).astype(np.float32)

        # 2. 绘制并保存热力图到前端静态目录
        plt.figure(figsize=(10, 8))
        plt.imshow(pressure_matrix, cmap=cm.jet)
        plt.colorbar()
        # plt.title(f"人员 {person} - 姿势 {posture_id}")

        # 按人员创建子目录（避免重名）
        person_image_dir = os.path.join(HEATMAP_IMAGE_DIR, person)
        os.makedirs(person_image_dir, exist_ok=True)
        image_name = f"posture_{posture_id}.png"
        image_path = os.path.join(person_image_dir, image_name)
        plt.savefig(image_path, bbox_inches="tight")
        plt.close()

        # 3. 拼接前端可访问的图片URL
        # 前端通过 /static/heatmap_images/[person]/[image_name] 访问
        relative_image_url = f"static/heatmap_images/{person}/{image_name}"
        return jsonify(
            {
                "code": 200,
                "data": {"image_url": relative_image_url},
                "msg": "图片生成成功",
            }
        )
    except Exception as e:
        return jsonify({"code": 500, "msg": f"生成图片失败：{str(e)}"})


# 新增接口4：允许前端直接访问生成的图片（Flask静态文件访问支持）
# 注：Flask默认会处理/static目录下的文件，此接口可省略，直接通过路径访问即可
@app.route("/static/heatmap_images/<path:filename>")
def serve_heatmap_image(filename):
    return send_from_directory(HEATMAP_IMAGE_DIR, filename)

# 新增：睡姿预测接口
@app.route("/api/predict_posture", methods=["POST"])
def predict_posture():
    if model is None:
        return jsonify({"code": 500, "msg": "模型未加载成功，无法预测"})
    
    data = request.json
    if not data or "matrix" not in data:
        return jsonify({"code": 400, "msg": "请求缺少压力矩阵数据（matrix字段）"})
    
    try:
        # 转换为numpy矩阵并校验形状
        pressure_matrix = np.array(data["matrix"], dtype=np.float32)
        if pressure_matrix.shape != (40, 26):
            return jsonify({"code": 400, "msg": f"压力矩阵形状错误，应为(40,26)，实际为{pressure_matrix.shape}"})
        
        # 调用预测函数
        predicted_id = predict_single_pressure_map(model, pressure_matrix)
        return jsonify({
            "code": 200,
            "data": {"predicted_posture_id": predicted_id},
            "msg": "预测成功"
        })
    except Exception as e:
        return jsonify({"code": 500, "msg": f"预测失败：{str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
