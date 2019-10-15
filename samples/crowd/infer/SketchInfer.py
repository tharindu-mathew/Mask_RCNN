import torch

from samples.crowd.infer import catmull, splinefit
import samples.crowd.infer.catmull

torch.cuda.current_device()
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import os
from loader import SketchDataSet
from PIL import Image
from infer import SketchSegment
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import samples.crowd.infer.splinefit


def get_resnet_model(resnet_type, pretrained):
    if resnet_type == '101':
        model_ft = models.resnet101(pretrained=pretrained)
    elif resnet_type == '18':
        model_ft = models.resnet18(pretrained=pretrained)
    elif resnet_type == '50':
        model_ft = models.resnet50(pretrained=pretrained)
    else:
        raise Exception("Unknown resenet type", resnet_type)
    return model_ft


class ClassifierInfer:
    def __init__(self, data_dir, resnet_type, use_cpu=False):
        self.use_cpu = use_cpu
        self.resnet_type = resnet_type
        self.class_names = None
        self.model_ft = None
        self.data_dir = data_dir
        self.save_name = os.path.basename(data_dir)
        self.device = None
        self.regress_data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def load_model(self):
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                  self.regress_data_transforms[x])
                          for x in ['train', 'val']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                      shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.class_names = image_datasets['train'].classes

        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        # plt.ion()   # interactive mode
        # imshow(out, title=[class_names for x in classes])

        model_ft = get_resnet_model(self.resnet_type, False)
        num_ftrs = model_ft.fc.in_features

        # setting number of params
        num_classes = len(self.class_names)
        print("class size : ", num_classes, self.class_names)
        output_layer_size = num_classes
        print("Setting output layer size to : ", output_layer_size)
        model_ft.fc = nn.Linear(num_ftrs, output_layer_size)

        model_ft = model_ft.to(device)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_ft = model_ft.to(self.device)
        model_name = os.path.join('model', self.save_name + '.pth');
        self.model_ft.load_state_dict(torch.load(model_name))
        self.model_ft.eval();

    def convert_output_id_to_class_name(self, outputs):
        _, preds = torch.max(outputs, 1)

        predicted_class_names = []
        for j in range(len(preds)):
            pred_class_name = self.class_names[preds[j]]
            predicted_class_names.append(pred_class_name)
        return predicted_class_names

    # def infer_imgs(self, img_numpy_arrs):
    #     input_list = []
    #     num_imgs = img_numpy_arrs.shape[0]
    #     for i in range(0, num_imgs):
    #         img_numpy = img_numpy_arrs[i]
    #         sample = Image.fromarray(img_numpy);
    #         input = self.regress_data_transforms['val'](sample)
    #         input_list.append(input)
    #         sample.save('test' + str(i) + '.png');
    #     inputs = torch.stack(input_list, dim=0)
    #     inputs = inputs.to(self.device)
    #     outputs = self.model_ft(inputs).cpu().detach().numpy()
    #     return outputs

    def infer_imgs_raw(self, img_tensors):
        inputs = img_tensors.to(self.device)
        outputs = self.model_ft(inputs)
        pred_class_names = self.convert_output_id_to_class_name(outputs)
        return pred_class_names

    # def infer_img(self, img_numpy):
    #     sample = Image.fromarray(img_numpy);
    #     sample.save('test.png');
    #     inputs = self.regress_data_transforms['val'](sample)
    #     inputs = inputs.unsqueeze(0)
    #     inputs = inputs.to(self.device)
    #     outputs = self.model_ft(inputs).cpu().detach().numpy()
    #     return outputs[0]

    # def infer_tensor(self, img_tensor):
    #     inputs = img_tensor.unsqueeze(0)
    #     inputs = img_tensor.to(self.device)
    #     outputs = self.model_ft(inputs).cpu().detach().numpy()
    #     return outputs[0]


class RegressInfer:
    def __init__(self, data_dir, resnet_type, use_cpu=False):
        self.use_cpu = use_cpu
        self.model_ft = None
        self.data_dir = data_dir
        self.save_name = os.path.basename(data_dir)
        self.resnet_type = resnet_type
        self.device = None
        self.regress_data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'infer': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def get_regress_transforms(self):
        return self.regress_data_transforms

    def load_model(self):
        image_datasets = {x: SketchDataSet.SketchDataSet('curve_params.csv', os.path.join(self.data_dir, x),
                                                         self.regress_data_transforms[x])
                          for x in ['train']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                      shuffle=True, num_workers=4)
                       for x in ['train']}

        inputs, curve_params = next(iter(dataloaders['train']))
        # model_ft = models.resnet18(pretrained=True)
        model_ft = get_resnet_model(self.resnet_type, False)
        num_ftrs = model_ft.fc.in_features

        # setting number of params
        print("Curve parameters size : ", curve_params.shape)
        output_layer_size = curve_params.shape[1]
        print("Setting output layer size to : ", output_layer_size)
        model_ft.fc = nn.Linear(num_ftrs, output_layer_size)

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if not self.use_cpu else "cpu")
        self.model_ft = model_ft.to(self.device)
        model_name = os.path.join('model', self.save_name + '.pth');
        print('loading model', self.save_name)
        if self.use_cpu:
            self.model_ft.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        else:
            self.model_ft.load_state_dict(torch.load(model_name))

        self.model_ft.eval();

    def infer_imgs(self, img_numpy_arrs):
        input_list = []
        num_imgs = img_numpy_arrs.shape[0]
        for i in range(0, num_imgs):
            img_numpy = img_numpy_arrs[i]
            sample = Image.fromarray(img_numpy);
            input = self.regress_data_transforms['val'](sample)
            input_list.append(input)
            sample.save('test' + str(i) + '.png');
        inputs = torch.stack(input_list, dim=0)
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs).cpu().detach().numpy()
        return outputs

    def infer_img_list(self, img_numpy_list):
        input_list = []
        num_imgs = len(img_numpy_list)
        for i in range(0, num_imgs):
            img_numpy = img_numpy_list[i]
            sample = Image.fromarray(img_numpy);
            input = self.regress_data_transforms['infer'](sample)
            input_list.append(input)
            sample.save('test' + str(i) + '.png');
        inputs = torch.stack(input_list, dim=0)
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs).cpu().detach().numpy()
        return outputs

    def infer_img(self, img_numpy):
        sample = Image.fromarray(img_numpy);
        inputs = self.regress_data_transforms['infer'](sample)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs).cpu().detach().numpy()
        return outputs[0]

    def infer_raw(self, img_tensor):
        inputs = img_tensor.unsqueeze(0)
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs).cpu().detach().numpy()
        return outputs[0]


class TotalInfer:
    # def __init__(self, base_dir, classifer_dir, regress_dirs, segment_dir, resnet_type, use_cpu=False):
    def __init__(self, args):
        self.args = args

        # print('hello')

        # self.classifier_infer = ClassifierInfer(os.path.join(base_dir, classifer_dir), resnet_type)
        # self.classifier_infer.load_model()
        #
        #
        # print(self.classifier_infer, self.regress_model_map)

    def init(self):
        self.segment_infer = SketchSegment.SketchSegment('logs', os.path.join(self.args.base_dir, self.args.segment_dir))
        self.segment_infer.load_model()

        self.data_dirs = self.args.regress_dirs
        self.regress_model_map = {}
        self.regress_data_transforms = None
        for regress_dir in self.args.regress_dirs:
            regress_infer = RegressInfer(os.path.join(self.args.base_dir, regress_dir), self.args.resnet_type, self.args.use_cpu)
            regress_infer.load_model()
            self.regress_model_map[regress_dir] = regress_infer
            if self.regress_data_transforms is None:
                self.regress_data_transforms = self.regress_model_map[regress_dir].get_regress_transforms()
        print(self.regress_model_map)
        self.regress_keys = self.regress_model_map.keys();


    def infer_imgs_with_classify(self, img_numpy_arrs):
        input_list = []
        num_imgs = img_numpy_arrs.shape[0]
        for i in range(0, num_imgs):
            img_numpy = img_numpy_arrs[i]
            sample = Image.fromarray(img_numpy);
            input = self.regress_data_transforms['val'](sample)
            input_list.append(input)
            sample.save('test' + str(i) + '.png');
        inputs = torch.stack(input_list, dim=0)
        classifer_outputs = self.classifier_infer.infer_imgs_raw(inputs)

        regress_outputs = []
        for i in range(0, num_imgs):
            assert (type(classifer_outputs[i]) == str)
            data_type = classifer_outputs[i]
            print('img ', i, ' is ', data_type)
            regress_output = self.regress_model_map[data_type].infer_raw(input_list[i])
            regress_outputs.append({data_type: regress_output.tolist()})

        return regress_outputs

    def contour_extraction_wall(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        kernel = np.ones((7, 7), np.uint8)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        cv2.imwrite('walls.png', img)
        contours = np.squeeze(contours)
        img_width = thresh.shape[0]
        contours = np.true_divide(contours, img_width)
        return contours;

    def contour_extraction_area(self, thresh):
        kernel_size = 5
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centers = []

        img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        for id, info in enumerate(hierarchy[0]):
            if info[2] == -1:
                M = cv2.moments(contours[id])

                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append(np.array((cX, cY)))
                cv2.drawContours(img, contours, id, (0, 255, 0), 1)

        img_width = thresh.shape[0]
        centers_np = np.stack(centers)
        centers_np = np.true_divide(centers_np, img_width)
        # c = np.mean(centers_np, axis=1)
        # d = np.std(centers_np, axis=1)

        # approxs = []
        # for cnt in contours:
        #     epsilon = 0.01 * cv2.arcLength(cnt, True)
        #     approx = cv2.approxPolyDP(cnt, epsilon, False)
        #     approxs.append(approx)

        # # #perimeter = cv2.arcLength(contours, True)
        # cv2.drawContours(img, approxs, -1, (0, 255, 0), 1)
        # # print(len(approxs))
        cv2.imwrite('areas.png', img)
        return centers_np

    def predict_line_segments(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        # blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # low_threshold = 50
        # high_threshold = 150
        # edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180 # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 20  # minimum number of pixels making up a line
        max_line_gap = 5  # maximum gap in pixels between connectable line segments
        # line_image = np.copy(img) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(thresh, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        # lines = cv2.HoughLines(thresh, rho, theta, threshold, np.array([]))

        print('num lines', len(lines))

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Draw the lines on the  image
        # lines_edges = cv2.addWeighted(img, 1.0, line_image, 1, 0)

        cv2.imwrite('walls.png', img)


    def colorline(self,
            x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
            linewidth=3, alpha=1.0):
        """
        http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
        http://matplotlib.org/examples/pylab_examples/multicolored_line.html
        Plot a colored line with coordinates x and y
        Optionally specify colors in the array z
        Optionally specify a colormap, a norm function and a line width
        """

        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))

        # Special case if a single number:
        if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
            z = np.array([z])

        z = np.asarray(z)

        segments = self.make_segments(x, y)
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                  linewidth=linewidth, alpha=alpha)

        ax = plt.gca()
        ax.add_collection(lc)

        return lc

    def make_segments(self, x, y):
        """
        Create list of line segments from x and y coordinates, in the correct format
        for LineCollection: an array of the form numlines x (points per line) x 2 (x
        and y) array
        """

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    def fit_curve_pts(self, estimated_pts):
        gray = cv2.imread('circular-path.png', 0)
        splinefit.fit_data(gray, estimated_pts)
        # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        # splinefit.fit_data(thresh, estimated_pts)


    def show_catmull_spline(self, points, area=False):
        c = catmull.create_catmull_spline(points)
        x, y = zip(*c)
        # plt.plot(x, y, c='blue')
        z = np.linspace(0, 1, len(x))
        self.colorline(x, y, z, cmap=plt.get_cmap('bwr'), linewidth=2)
        plt.show()

    def segment_img(self, img_numpy):
        mask_dict = self.segment_infer.segment_imgV2(img_numpy)
        return mask_dict

    def regress_imgs(self, label_and_imgs):
        regress_outputs = {}
        i = 0
        for data_type in label_and_imgs.keys():
            all_imgs = label_and_imgs[data_type]
            for img in all_imgs:
                for key in self.regress_keys:
                    if key.startswith(data_type):
                        sample = Image.fromarray(img);
                        sample.save('regress' + str(data_type) + str(i) + '.png')
                        i += 1
                        regress_output = self.regress_model_map[key].infer_img(img)
                        points = regress_output.reshape((4, 2))
                        if data_type not in regress_outputs:
                            regress_outputs[data_type] = []
                        if data_type.startswith('area'):
                            regress_outputs[data_type].append({"control_points": regress_output.tolist()})
                            self.show_catmull_spline(points, True)
                        else:
                            # regress_output = splinefit.fit_data(thresh, regress_output.reshape((4, 2)))
                            regress_outputs[data_type].append({"control_points": regress_output.tolist()})
                            self.show_catmull_spline(points, False)
        return regress_outputs

    def infer_imgs(self, img_numpy):
        result = self.segment_infer.segment_img(img_numpy)

        regress_outputs = []
        i = 0
        for data_type in result.keys():
            all_imgs = result[data_type]
            for img, roi in all_imgs:
                if data_type.startswith('wall'):
                    # wall_line_segments = self.predict_line_segments(img)
                    wall_contours = self.contour_extraction_wall(img)
                    regress_outputs.append({data_type: wall_contours.tolist()})

                else:
                    for key in self.regress_keys:
                        if key.startswith(data_type):
                            # scale image before infer
                            # h, w, c = img.shape
                            # y1, x1, y2, x2 = roi
                            #
                            # roi_img = img[y1:y2, x1:x2, :]
                            #
                            # dest_img_w = 224
                            # src_img_w = 210
                            # img_offset = (dest_img_w - src_img_w)// 2
                            # scaled_img_224 = np.ones((dest_img_w, dest_img_w, 3)).astype(np.uint8) * 255
                            # scaled_roi = cv2.resize(roi_img, (src_img_w, src_img_w))
                            # scaled_img_224[img_offset:img_offset+src_img_w, img_offset:img_offset+src_img_w, :] = scaled_roi
                            # cv2.imwrite('scaled_img' + str(i) + data_type + '.png', scaled_img_224)
                            # i += 1
                            # regress_output = self.regress_model_map[key].infer_img(scaled_img_224)

                            regress_output = self.regress_model_map[key].infer_img(img)

                            # ratio = 224.0/256.0
                            # regress_output *= ratio
                            # regress_output = splinefit.fit_data(img, regress_output.reshape((4, 2)))
                            points = regress_output.reshape((4, 2))
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

                            if (data_type.startswith('area')):
                                blob_centers = self.contour_extraction_area(thresh)
                                regress_outputs.append({data_type: {"control_points" : regress_output.tolist(), "blobs" : blob_centers.tolist(), "roi": roi.tolist() }})
                                self.show_catmull_spline(points, True)
                            else:
                                # regress_output = splinefit.fit_data(thresh, regress_output.reshape((4, 2)))
                                regress_outputs.append({data_type: {"control_points": regress_output.tolist(), "roi": roi.tolist() }})
                                self.show_catmull_spline(points, False)

        return regress_outputs
