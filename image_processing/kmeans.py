import numpy as np
import cv2
import timeit
from flask import request
import urllib.request

self_host = 'http://localhost:5000'


def run():
    url = request.args.get('url')
    if url is None:
        url = 'http://localhost:5000/images?name=home.jpg'

    rs = {
        'results': []
    }

    try:
        req = urllib.request.urlopen(url)
        img_name = url.split('=')[1].split('.')[0]
        img_ext = url.split('=')[1].split('.')[1]
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        # Ks = [2, 8, 32, 64, 256]
        Ks = [2, 4, 8, 16, 32]

        Z = img.reshape((-1, 3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        for K in Ks:
            start = timeit.default_timer()

            ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

            print(ret)
            print(label)
            print(center)

            # Now convert back into uint8, and make original image
            center = np.uint8(center)
            res = center[label.flatten()]
            print('-------------')
            print(res)
            print('-------------')
            res2 = res.reshape(img.shape)

            print('+++++++++++')
            print(res2)
            print('+++++++++++')

            # write file
            cv2.imwrite('D:/ai/src/case_study/image_processing/images/{}-{}.{}'.format(img_name, K, img_ext), res2)

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
        rs['message'] = e.message
        return rs


# if __name__ == "__main__":
#     run()
