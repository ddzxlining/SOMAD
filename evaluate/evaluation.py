import argparse
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from collections import OrderedDict
import pickle

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from skimage import morphology
from skimage.segmentation import mark_boundaries
from evaluate.utils import per_pixel_pro_and_roc_auc_score


class Evaluator:
    def __init__(self, args):
        self.save_dir=os.path.join(args.save_path,'evaluation',args.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.fig, ax = plt.subplots(1, 3, figsize=(20, 10))
        self.fig_img_rocauc = ax[0]
        self.fig_pixel_rocauc = ax[1]
        self.fig_pro_auc = ax[2]
        self.thresholds = OrderedDict()

        self.total_image_roc_auc = []
        self.total_pixel_roc_auc = []
        self.total_pro_auc = []

        self.total_image_fpr = []
        self.total_image_tpr = []

        self.total_pixel_fpr = []
        self.total_pixel_tpr = []

        self.total_pro_fpr = []
        self.total_pro_pro=[]

        if self.can_resume():
            self.resume()

    def can_resume(self):
        return os.path.isfile(os.path.join(self.save_dir, 'result.pkl'))

    def resume(self):
        with open(os.path.join(self.save_dir, 'result.pkl'), 'rb') as f:
            result = pickle.load(f)

        self.thresholds = result['thresholds']

        self.total_pixel_roc_auc = result['pixel_roc']
        self.total_image_roc_auc = result['image_roc']
        self.total_pro_auc = result['pro_auc']

        self.total_image_fpr = result['image_fpr']
        self.total_image_tpr = result['image_tpr']

        self.total_pixel_fpr = result['pixel_fpr']
        self.total_pixel_tpr = result['pixel_tpr']

        self.total_pro_fpr=result['pro_fpr']
        self.total_pro_pro = result['pro_pro']

    def store(self):
        result = {}
        result['thresholds'] = self.thresholds

        result['pixel_roc'] = self.total_pixel_roc_auc
        result['image_roc'] = self.total_image_roc_auc
        result['pro_auc'] = self.total_pro_auc

        result['image_fpr'] = self.total_image_fpr
        result['image_tpr'] = self.total_image_tpr

        result['pixel_fpr'] = self.total_pixel_fpr
        result['pixel_tpr'] = self.total_pixel_tpr

        result['pro_fpr']=self.total_pro_fpr
        result['pro_pro'] = self.total_pro_pro

        with open(os.path.join(self.save_dir, 'result.pkl'), 'wb') as f:
            pickle.dump(result, f)

    def eval_class(self, class_name, gt_list, gt_mask_list, img_scores, pixel_scores):
        """
        @param class_name: evaluted data class name.
        @param gt_list: image-level-gt shape: n,
        @param gt_mask_list: pixel-level-gt n,H,W
        @param img_scores: image-level-scores shape: n,
        @param pixel_scores: pixel-level-scores shape: n,H,W
        """
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        threshold = self.compute_thresh(gt_mask, pixel_scores)
        self.thresholds[class_name] = threshold

        # calculate image-pixel level ROCAUC
        image_fpr, image_tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)

        self.total_image_roc_auc.append(img_roc_auc)

        self.total_image_fpr.append(image_fpr)
        self.total_image_tpr.append(image_tpr)

        # calculate per-region overlap auc.
        per_pixel_rocauc, per_pixel_fpr, per_pixel_tpr, pro_auc, sub_fpr, sub_pro = per_pixel_pro_and_roc_auc_score(
            gt_mask.flatten(), gt_mask.shape, pixel_scores.flatten(), max_fpr=0.3)

        self.total_pixel_roc_auc.append(per_pixel_rocauc)
        self.total_pro_auc.append(pro_auc)

        self.total_pixel_fpr.append(per_pixel_fpr)
        self.total_pixel_tpr.append(per_pixel_tpr)

        self.total_pro_fpr.append(sub_fpr)
        self.total_pro_pro.append(sub_pro)

        self.store()

        print('class name: %s' % class_name)
        print('  image ROCAUC: %.3f' % (img_roc_auc))
        print('  pixel ROCAUC: %.3f' % (per_pixel_rocauc))
        print('  PRO_AUC: %.3f' % pro_auc)

    def compute_thresh(self, gt_mask, pixel_scores):
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), pixel_scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]
        return threshold

    def save(self):
        self.store()
        print('---------------------------------------------------------------------------------------')
        for idx, class_name in enumerate(list(self.thresholds.keys())):
            print('class name: %10s ,image ROCAUC: %.3f ,pixel ROCAUC: %.3f ,PRO_AUC: %.3f' % (
            class_name, self.total_image_roc_auc[idx], self.total_pixel_roc_auc[idx], self.total_pro_auc[idx]))
            self.fig_img_rocauc.plot(self.total_image_fpr[idx], self.total_image_tpr[idx],
                                     label='%s img_ROCAUC: %.3f' % (class_name, self.total_image_roc_auc[idx]))
            self.fig_pixel_rocauc.plot(self.total_pixel_fpr[idx], self.total_pixel_tpr[idx],
                                       label='%s ROCAUC: %.3f' % (class_name, self.total_pixel_roc_auc[idx]))
            self.fig_pro_auc.plot(self.total_pro_fpr[idx], self.total_pro_pro[idx], label='%s PROAUC: %.3f' % (class_name, self.total_pro_auc[idx]))
        print('---------------------------------------------------------------------------------------')
        print()
        print('Average Image ROCAUC: %.3f' % np.mean(self.total_image_roc_auc))
        self.fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(self.total_image_roc_auc))
        self.fig_img_rocauc.legend(loc="lower right")

        print('Average pixel ROCAUC: %.3f' % np.mean(self.total_pixel_roc_auc))
        self.fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(self.total_pixel_roc_auc))
        self.fig_pixel_rocauc.legend(loc="lower right")

        print('Average PROAUC: %.3f' % np.mean(self.total_pro_auc))
        self.fig_pro_auc.title.set_text('Average PROAUC: %.3f' % np.mean(self.total_pro_auc))
        self.fig_pro_auc.legend(loc="lower right")

        self.fig.tight_layout()
        self.fig.savefig(os.path.join(self.save_dir, 'roc_curve.png'))

    def load_result(self):
        if not self.can_resume():
            print('result.pkl not found!')
            return
        result = pickle.load(open(os.path.join(self.save_dir, 'result.pkl'), 'rb'))
        self.thresholds = result['thresholds']
        self.total_pixel_roc_auc = result['pixel_roc']
        self.total_image_roc_auc = result['image_roc']
        self.total_pro_auc = result['pro_auc']
        print('---------------------------------------------------------------------------------------')
        for idx, (k, v) in enumerate(self.thresholds.items()):
            print('class name: %10s ,image ROCAUC: %.3f ,pixel ROCAUC: %.3f ,PRO_AUC: %.3f' % (k,self.total_image_roc_auc[idx],self.total_pixel_roc_auc[idx],self.total_pro_auc[idx]))
        print('---------------------------------------------------------------------------------------')
        print()
        print('Average Image ROCAUC: %.3f' % np.mean(self.total_image_roc_auc))
        print('Average pixel ROCAUC: %.3f' % np.mean(self.total_pixel_roc_auc))
        print('Average PROAUC: %.3f' % np.mean(self.total_pro_auc))

        return self.thresholds, self.total_image_roc_auc, self.total_pixel_roc_auc, self.total_pro_auc

    def visualize_class(self, class_name, test_imgs, pixel_scores, gt_mask):
        if class_name not in self.thresholds:
            self.thresholds[class_name] = self.compute_thresh(gt_mask, pixel_scores)
        self._plot_fig(test_imgs, pixel_scores, gt_mask, self.thresholds[class_name], class_name)

    def _plot_fig(self, test_img, scores, gts, threshold, class_name):
        num = len(scores)
        vmax = scores.max() * 255.
        vmin = scores.min() * 255.
        for i in range(num):
            img = test_img[i]
            img = denormalization(img)
            gt = gts[i].transpose(1, 2, 0).squeeze()
            heat_map = scores[i] * 255
            mask = scores[i]
            mask[mask > threshold] = 1
            mask[mask <= threshold] = 0
            kernel = morphology.disk(4)
            mask = morphology.opening(mask, kernel)
            mask *= 255
            vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
            fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
            fig_img.subplots_adjust(right=0.9)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Image')
            ax_img[1].imshow(gt, cmap='gray')
            ax_img[1].title.set_text('GroundTruth')
            ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
            ax_img[2].imshow(img, cmap='gray', interpolation='none')
            ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
            ax_img[2].title.set_text('Predicted heat map')
            ax_img[3].imshow(mask, cmap='gray')
            ax_img[3].title.set_text('Predicted mask')
            ax_img[4].imshow(vis_img)
            ax_img[4].title.set_text('Segmentation result')
            left = 0.92
            bottom = 0.15
            width = 0.015
            height = 1 - 2 * bottom
            rect = [left, bottom, width, height]
            cbar_ax = fig_img.add_axes(rect)
            cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
            cb.ax.tick_params(labelsize=8)
            font = {
                'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 8,
            }
            cb.set_label('Anomaly Score', fontdict=font)
            save_dir = os.path.join(self.save_dir, 'visualize', class_name)
            os.makedirs(save_dir, exist_ok=True)
            fig_img.savefig(os.path.join(save_dir, '{}'.format(i)), dpi=100)
            plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument("--data_path", type=str, default="D:/datasets/mvtec_anomaly_detection")
    parser.add_argument("--save_path", type=str, default="../mvtec_result")
    parser.add_argument("--arch", type=str, choices=['resnet18, wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    eval = Evaluator(args)
    eval.load_result()
    print()
