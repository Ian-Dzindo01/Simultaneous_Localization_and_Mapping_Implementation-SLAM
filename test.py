#!/usr/bin/python3

import numpy as np
import cv2

W = 1920//2
H = 1080//2


if __name__ == "__main__":
    cap = cv2.VideoCapture('video/test1.mp4')

    if (cap.isOpened()== False):
      print("Error opening video stream or file")

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = np.mean(frame, axis=2).astype(np.uint8)

        if ret == True:
            cv2.imshow('Frame',frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()

    cv2.destroyAllWindows()


def extract(img):
    # detection
    feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)      # strong corners on an image
                                                                                                                        # image needs to be black and white
    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]           # extract keypoints from features

    kps, des = self.orb.compute(img, kps)                                        # compute keypoints and descriptors

    # matching
    ret = []
    if self.last is not None:
        matches = self.bf.knnMatch(des, self.last['des'], k=2)                         # using knn instead of regular match

        # ratio test
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                kp1 = kps[m.queryIdx].pt
                kp2 = self.last['kps'][m.trainIdx].pt
                ret.append((kp1, kp2))


    # outlier inlier filtering using fundamental matrices
    if len(ret) > 0:
        ret = np.array(ret)


        # normalize coordinates: subtract to move to 0
        ret[: ,0 , :] = np.dot(self.Kinv, add_ones(ret[:, 0 ,:]).T).T[:, 0:2]
        ret[: ,1 , :] = np.dot(self.Kinv, add_ones(ret[:, 1 ,:]).T).T[:, 0:2]

        model, inliers = ransac((ret[:,0], ret[:,1]),
                                FundamentalMatrixTransform,
                                min_samples=8,
                                residual_threshold=1,
                                max_trials=100)

        ret = ret[inliers]       # outlier removal

        # s,v,d = np.linalg.svd(model.params)     # singular value decomposition on matrix
        # print(v)

    self.last = {'kps' : kps, 'des' : des}     # keypoints and descriptors
    return ret

K = np.array(([1,0,W//2], [0,1,H//2], [0,0,1]))    # intrinsic matrix
Kinv = np.linalg.inv(K)

# K = add_ones(K.T).T
# print(K)
# print(K[:, 0:2])

