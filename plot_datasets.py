import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import astropy.io.fits as pf
import pylab
# from plot_coords import allskymap,coords
import plot_coords.coords as coords
import plot_coords.allskymap as allskymap
# from plot_coords.allskymap import Allskymap



def create_map():
    # Note that Hammer & Mollweide projections enforce a 2:1 aspect ratio.
    # Use figure size good for a 2:1 plot.
    fig = pylab.figure()
    fig.set_size_inches(12, 6)
    pylab.clf()
    # Set up the projection and draw a grid.
    m = allskymap.AllSkyMap(projection = 'hammer', lon_0 = 180)
    # Save the bounding limb to use as a clip path later.
    limb = m.drawmapboundary(fill_color = 'white')
    m.drawparallels(np.arange(-75, 75, 15), linewidth = 0.5, dashes = [1, 2],
                    labels = [1, 0, 0, 0], fontsize = 12)
    m.drawmeridians(np.arange(30, 331, 30), linewidth = 0.5, dashes = [1, 2])
    # m.drawmeridians(arange(-150,151,30), linewidth=0.5, dashes=[1,2])

    # Label a subset of meridians.
    # lons = np.arange(0,360,30)
    lons = np.arange(30, 331, 30)
    # lons = arange(-150,151,30)
    m.label_meridians(lons, fontsize = 12, vnudge = 1, halign = 'left',
                      hnudge = -1)  # hnudge<0 shifts to right
    # plot b=0
    for nb in np.array([-60, -30, 0, 30, 60]):
        l = np.arange(0, 361, 1)
        b = np.ones(len(l)) * nb
        ll, bb = [], []
        j = -1
        last = -1
        for i in range(len(l)):
            x, y = coords.eq2gal(l[i], b[i])
            if i > 0 and abs(last - x) > 100:
                j = i
            last = x

            x, y = m(x, y)
            ll.append(x)
            bb.append(y)
        ll = ll[:j][::-1] + ll[j:][::-1]
        bb = bb[:j][::-1] + bb[j:][::-1]
        pylab.plot(ll, bb, 'k', linewidth = 1.0, alpha = 0.5)  # ,label='eq dec=0')

    x1, y1 = coords.eq2gal(np.array([0, 0, 98, 270, 90, 260]), np.array([-90, 90, -60, 60, -30, 30]))
    x1, y1 = m(x1, y1)
    pylab.scatter(x1[0], y1[0], c = 'k', s = 5, marker = 'o')
    pylab.scatter(x1[1], y1[1], c = 'k', s = 5, marker = 'o')
    pylab.text(x1[0], y1[0], 'SP')
    pylab.text(x1[1], y1[1], 'NP')
    pylab.text(x1[2], y1[2], allskymap.angle_symbol(-60))
    pylab.text(x1[3], y1[3], allskymap.angle_symbol(60))
    pylab.text(x1[4], y1[4], allskymap.angle_symbol(-30))
    pylab.text(x1[5], y1[5], allskymap.angle_symbol(30))
    return m


class st:
    def __init__(self):
        self.m = create_map()

    # def get_data(self, ra,dec):
    #     self.ra = ra
    #     self.dec = dec

    def go_plot(self):
        label = self.title
        plt.scatter(self.ra, self.dec, s = 0.2, edgecolor = self.facecolor, facecolor = self.facecolor,
                    label = label)

    def plot_lamost(self, lamost_ra, lamost_dec):
        # self.get_data('/home/wangr/data/plot_for_hw/apogee_lamost_radec.txt')
        ra, dec = coords.eq2gal(lamost_ra, lamost_dec)
        self.ra, self.dec = self.m(ra, dec)
        self.facecolor = 'grey'
        self.title = 'LAMOST-II'
        # self.hatch='\\\\'
        self.go_plot()

    def plot_apogee(self, apogee_ra, apogee_dec):
        # self.get_data('/home/wangr/data/plot_for_hw/apogee.csv')
        # self.ra=np.array([180])
        # self.dec=np.array([0])
        ra, dec = coords.eq2gal(apogee_ra, apogee_dec)
        self.ra, self.dec = self.m(ra, dec)
        # self.get_data(apogee_ra, apogee_dec)
        self.facecolor = 'red'
        self.title = 'APOGEE'
        # self.hatch='\\\\'
        self.go_plot()


def plot_med_apogee():
    path_lamost = '/home/wangr/data/LAMOST/medres_spec/medres_spec_info_all.csv'
    data_lamost = pd.read_csv(path_lamost)
    lamost_ra = data_lamost.objra.values
    lamost_dec = data_lamost.objdec.values

    # path_apogee = '/home/wangr/data/catalogues/APOGEE.fits'
    # hdu = pf.open(path_apogee)
    # data_apogee = hdu[1].data
    # data_apogee_ra = data_apogee.RA
    # data_apogee_dec = data_apogee.DEC
    # print data_apogee_ra.shape
    # print data_apogee_dec.shape

    obj = st()
    obj.plot_lamost(lamost_ra, lamost_dec)
    # obj.plot_apogee(data_apogee_ra, data_apogee_dec)
    plt.legend()
    plt.title('Galactic Coordinates')
    # plt.savefig('med_apogee_lasp_radec.png')
    plt.show()


def plot_param_distribution(trainset, testset):
    sns.set(style = "white", palette = "muted", color_codes = True)

    f, axes = plt.subplots(2, 2)

    # sns.despine(left = True)
    # sns.distplot(trainset.SNR, hist=False, color="lime", kde_kws={"shade": True}, ax=axes[0, 0])
    # sns.distplot(testset.SNR, hist=False, color="mediumorchid", kde_kws={"shade": True}, ax=axes[0, 0])
    #
    # sns.distplot(trainset.rv, hist = False, color = "g", kde_kws = {"shade": True}, ax = axes[0, 0])
    # sns.distplot(testset.rv, hist = False, color = "r", kde_kws = {"shade": True}, ax = axes[0, 0])

    sns.distplot(trainset.teff, hist = False, color = "lime", kde_kws = {"shade": False}, ax = axes[0, 0],
                 label = 'training sets',axlabel = 'Teff')
    sns.distplot(testset.teff, hist = False, color = "mediumorchid", kde_kws = {"shade": False},
                 ax = axes[0, 0], label = 'test sets',axlabel = 'Teff')
    # plt.xlabel('Teff')

    sns.distplot(trainset.logg, hist = False, color = "lime", kde_kws = {"shade": False}, ax = axes[0, 1],
                 label = 'training sets',axlabel = 'logg')
    sns.distplot(testset.logg, hist = False, color = "mediumorchid", kde_kws = {"shade": False},
                 ax = axes[0, 1], label = 'test sets',axlabel = 'logg')
    # plt.xlabel('log g')

    sns.distplot(trainset.mh, hist = False, color = "lime", kde_kws = {"shade": False}, ax = axes[1, 0],
                 label = 'training sets',axlabel = '[M/H]')
    sns.distplot(testset.mh, hist = False, color = "mediumorchid", kde_kws = {"shade": False}, ax = axes[1, 0],
                 label = 'test sets',axlabel = '[M/H]')
    # plt.xlabel('[M/H]')

    sns.distplot(trainset.am, hist = False, color = "lime", kde_kws = {"shade": False}, ax = axes[1, 1],
                 label = 'training sets')
    sns.distplot(testset.am, hist = False, color = "mediumorchid", kde_kws = {"shade": False}, ax = axes[1, 1],
                 label = 'test sets')
    plt.xlabel('[$\\alpha$/M]')

    plt.legend()

    plt.show()


if __name__ == '__main__':
    plot_med_apogee()
    # trainset_raw_y = np.load('/home/wangr/data/LAMOST/dnn_test_sets/ytrain_22.npy')
    # testset_raw_y = np.load('/home/wangr/data/LAMOST/dnn_test_sets/ytest_22.npy')
    # trainset_param = pre_model.denormalize(trainset_raw_y)
    # testset_param = pre_model.denormalize(testset_raw_y)
    # train = pd.DataFrame(columns = ['teff', 'logg', 'mh', 'am'], data = trainset_param)
    # test = pd.DataFrame(columns = ['teff', 'logg', 'mh', 'am'], data = testset_param)
    # plot_param_distribution(train, test)
