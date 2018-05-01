from .VidStab import VidStab

import cv2
import numpy as np
from progress.bar import IncrementalBar
import imutils.feature.factories as kp_factory


class Stitcher(VidStab):

    def kp_detect_describe(self, gray):
        # detect keypoints in the image
        detector = kp_factory.FeatureDetector_create(self.kp_method)
        kps = detector.detect(gray)

        # extract features from the image
        extractor = kp_factory.DescriptorExtractor_create("SIFT")
        (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return kps, features

    @staticmethod
    def match_kps(kps_a, kps_b, features_a, features_b,
                  ratio=0.75, reproj_thresh=4.0):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(features_a, features_b, 2)
        matches = []

        # loop over the raw matches
        for m in raw_matches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            pts_a = np.float32([kps_a[i] for (_, i) in matches])
            pts_b = np.float32([kps_b[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (homography, status) = cv2.findHomography(pts_a, pts_b, cv2.RANSAC,
                                                      reproj_thresh)

            # return the matches along with the homography matrix
            # and status of each matched point
            return matches, homography, status

        # otherwise, no homography could be computed
        return None

    def gen_stitching(self, input_path, max_frames=None, border_size=200, show_progress=True):
        """Generate frame transformations to apply for stabilization

        :param input_path: Path to input video to stabilize.
        Will be read with cv2.VideoCapture; see opencv documentation for more info.
        :param border_size:
        :param show_progress: Should a progress bar be displayed to console?
        :return: Nothing is returned.  The results are added as attributes: trajectory, smoothed_trajectory, & transforms
        """
        # set up video capture
        vid_cap = cv2.VideoCapture(input_path)
        frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is None:
            max_frames = frame_count

        # read first frame
        _, prev_frame = vid_cap.read()
        h, w = prev_frame.shape[:2]

        # convert to gray scale
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        stitched = cv2.copyMakeBorder(prev_frame,
                                      top=border_size,
                                      bottom=border_size,
                                      left=border_size,
                                      right=border_size,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])
        # detect keypoints
        prev_kps, prev_kp_descs = self.kp_detect_describe(prev_frame_gray)

        if show_progress:
            bar = IncrementalBar('Generating Transforms', max=max_frames, suffix='%(percent)d%%')
        # iterate through frame count
        for i in range(frame_count - 1):
            if i > max_frames:
                break
            # read current frame
            _, cur_frame = vid_cap.read()
            # convert to gray
            cur_frame_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            # detect keypoints
            cur_kps, cur_kp_descs = self.kp_detect_describe(cur_frame_gray)

            match_result = self.match_kps(prev_kps, cur_kps, prev_kp_descs, cur_kp_descs)

            if match_result is not None:
                matches, homography, status = match_result
                bordered_frame = cv2.copyMakeBorder(cur_frame,
                                                    top=border_size,
                                                    bottom=border_size,
                                                    left=border_size,
                                                    right=border_size,
                                                    borderType=cv2.BORDER_CONSTANT,
                                                    value=[0, 0, 0])
                warped = cv2.warpPerspective(bordered_frame, homography,
                                             (w + 2 * border_size, h + 2 * border_size))

                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                _, threshed = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY_INV)
                threshed = cv2.dilate(threshed, None, iterations=2)

                masked = cv2.bitwise_and(stitched, stitched, mask=threshed)

                stitched = np.maximum(masked, warped)
            else:
                raise ValueError('No matching points to stitch')

            # set current frame to prev frame for use in next iteration
            prev_kps = cur_kps[:]
            prev_kp_descs = cur_kp_descs[:]
            if show_progress:
                bar.next()
        bar.finish()

        return stitched
