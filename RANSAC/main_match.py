import hw_utils as utils
import matplotlib.pyplot as plt


def main():
    # No RANCAC
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/basmati', ratio_thres=0.6)
    plt.title('Match')
    plt.imshow(im)

    # scene, book
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/book', ratio_thres=0.7)
    plt.title('Match')
    plt.imshow(im)
    
    # scene, box
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/box', ratio_thres=0.7)
    plt.title('Match')
    plt.imshow(im)



    # With RANCAC
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/scene', './data/basmati',
        ratio_thres=0.8, orient_agreement=20, scale_agreement=0.5)
    plt.title('MatchRANSAC')
    plt.imshow(im)

    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/basmati', ratio_thres=0.6)
    plt.title('Match')
    plt.imshow(im)

    # Library-Library2
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/library', './data/library2',
        ratio_thres=0.8, orient_agreement=10, scale_agreement=0.5)

    plt.title('MatchRANSAC')
    plt.imshow(im)

    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/library', './data/library2', ratio_thres=0.8)
    plt.title('Match')
    plt.imshow(im)

if __name__ == '__main__':
    main()
