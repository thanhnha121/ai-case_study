import numpy as np
import cv2
import timeit
# from flask import request
import urllib.request
from numpy import array

self_host = 'http://localhost:5000'


def run():
    inp = []
    for x in range(150):
        x = x + 1
        if x < 10:
            x = '00' + str(x)
        elif 10 <= x < 100:
            x = '0' + str(x)

        i = cv2.imread('images/handwriting/image_part_{}.jpg'.format(x))
        if i is None:
            continue

        tmp = i.reshape((-1, 1))
        tmp = list(tmp)
        for t in range(len(tmp)):
            tmp[t] = list(tmp[t])

        inp.append(tmp)

    inp = array(inp).reshape((-1, 4563))

    Z = np.float32(inp)

    K = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .1)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 2, cv2.KMEANS_PP_CENTERS)

    print(ret)
    print('---------------')
    print(label)
    print('---------------')
    print(center)
    print('---------------')

    for x in range(len(center)):
        c = np.uint8(center[x])
        res2 = c.reshape(39, 39, 3)
        cv2.imshow(str(x), res2)

    cv2.waitKey(0)

    return

    try:
        req = urllib.request.urlopen(url)
        img_name = url.split('=')[1].split('.')[0]
        img_ext = url.split('=')[1].split('.')[1]
        # img_ext = 'jpg'
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        # print(img)

        # Ks = [2, 8, 32, 64, 128, 256]
        Ks = [2, 4, 6, 8, 16, 32, 64]
        # Ks = [2, 4, 8, 16, 32, 64, 256]

        Z = img.reshape((-1, 1))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)

        for K in Ks:
            start = timeit.default_timer()

            ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

            print('/////////////')
            print(K)
            print(ret)
            print(label)
            print(center)

            # Now convert back into uint8, and make original image
            center = np.uint8(center)
            res = center[label.flatten()]
            print('-------------')
            print(res)
            res2 = res.reshape(img.shape)

            print('+++++++++++++')
            print(res2)

            # write file
            cv2.imwrite(
                'E:/ai/case_study/image_processing/images/{}-{}.{}'.format(img_name, K, img_ext),
                res2,
                [cv2.IMWRITE_JPEG_QUALITY, 90]
            )

            # handle return
            rs['results'].append({
                'time': timeit.default_timer() - start,
                'url': self_host + '/images?name={}-{}.{}'.format(img_name, K, img_ext),
                'k': K
            })

        rs['status'] = 1
        rs['message'] = 'success'
        return rs

    except Exception as e:
        rs['results'] = []
        rs['status'] = 0
        rs['message'] = str(e)
        return rs


if __name__ == "__main__":
    run()
