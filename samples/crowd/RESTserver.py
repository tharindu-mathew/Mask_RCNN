from flask_restful import Resource, reqparse
from flask import Flask, request, jsonify
from infer import SketchInfer
import numpy as np
import argparse
import os

users = [
    {
        "name": "Nicholas",
        "age": 42,
        "occupation": "Network Engineer"
    },
    {
        "name": "Elvin",
        "age": 30,
        "occupation": "Doctor"
    },
    {
        "name": "Jass",
        "age": 22,
        "occupation": "Web developer"
    }
]

app = Flask(__name__)


@app.route('/imginfer', methods=['GET', 'POST'])
def infer():
    content = request.json
    img_numpy = np.array(content['imgs']).astype(np.uint8)
    # output_params = inference_engine.infer_img(img_numpy)
    output_params = inference_engine.infer_imgs(img_numpy)
    print(output_params)
    return jsonify({"params": output_params})


class User(Resource):
    def get(self, name):
        for user in users:
            if (name == user["name"]):
                return user, 200
        return "User not found", 404

    def post(self, name):
        parser = reqparse.RequestParser()
        parser.add_argument("age")

class ImgInfer(Resource):
    def post(self):
        #print(request.json)
        parser = reqparse.RequestParser()
        parser.add_argument("img")
        args = parser.parse_args()
        print(args)
        return "Success", 200

def main():
    global inference_engine
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--classification_dir', type=str, required=True)
    parser.add_argument('--regress_dirs', nargs='+', default=[])
    parser.add_argument('--segment_dir', type=str, required=True)
    parser.add_argument('--resnet_type', type=str, required=True)
    parser.add_argument('--use_cpu', action='store_true')

    args = parser.parse_args()
    os.chdir(os.path.dirname(__file__))

    print('current working dir', os.getcwd())

    inference_engine = SketchInfer.TotalInfer(args)
    # inference_engine.init()
    # app.run()

    estimated_pts = np.array([0.6785781979560852, 0.7342093586921692, 0.556641697883606, 0.5983264446258545, 0.44624197483062744, 0.47519931197166443, 0.3184840679168701, 0.3628543019294739])
    estimated_pts = np.reshape(estimated_pts, (estimated_pts.shape[0] // 2, 2))

    inference_engine.fit_curve_pts(estimated_pts)

    # inference_engine.show_catmull_spline(np.random.rand(20).reshape((10, 2)))
    # inference_engine.show_catmull_spline(np.random.rand(6,2), True)
    #inference_engine.load_model();
    # api = Api(app)
    # api.add_resource(User, "/user/<string:name>")
    # api.add_resource(ImgInfer, "/imginfer")
    #app.run(debug=True)



if __name__ == '__main__':
    main()
    # import cv2
    # img = cv2.imread("test0area.png")
    # # inference_engine.predict_line_segments(img)
    # inference_engine.contour_extraction_area(img)

