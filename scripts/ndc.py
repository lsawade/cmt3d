# %%
import typing as tp
from cmt3d.viz import utils
from cmt3d.viz import catalog_utils as cutils
from cmt3d.viz import catalog_plot_utils as cputils
from cmt3d.cmt_catalog import CMTCatalog
from cmt3d.viz.compare_catalogs import CompareCatalogs
from glob import glob
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict

utils.updaterc()
plt.ion()

# %%
# Load the CMT catalog
cat1 = CMTCatalog.from_file_list(glob("events/gcmt3d_fix/*"))


# %%


cat2 = deepcopy(cat1)
cutils.make_catalog_dc(cat2)

# %%

pdict = dict(
    m_rt=0.15,
    m_rp=0.3,
    m_tp=0.1,
    m_rr=0.15,
    # m_tt=0.1,
    # m_pp=0.1,
)


# %%

mech = cat1.getvals(vtype="mtype")
idx_mech = np.where(mech == "strike-slip")[0]
M0 = cat1.getvals(vtype="M0")[idx_mech]
Mrr = cat1.getvals(vtype="m_rr")[idx_mech] / M0
Mrt = cat1.getvals(vtype="m_rt")[idx_mech] / M0
Mrp = cat1.getvals(vtype="m_rp")[idx_mech] / M0
idx_mech = np.where(mech == "normal")[0]
M0 = cat1.getvals(vtype="M0")[idx_mech]
Mtt = cat1.getvals(vtype="m_tt")[idx_mech] / M0
Mpp = cat1.getvals(vtype="m_pp")[idx_mech] / M0
Mtp = cat1.getvals(vtype="m_tp")[idx_mech] / M0


(
    mmrr,
    mmrt,
    mmrp,
    mmtt,
    mmpp,
) = (
    np.mean(Mrr),
    np.mean(Mrt),
    np.mean(Mrp),
    np.mean(Mtt),
    np.mean(Mpp),
)
smrr, smrt, smrp, smtt, smpp = (
    np.std(Mrr),
    np.std(Mrt),
    np.std(Mrp),
    np.std(Mtt),
    np.std(Mpp),
)


# %%


meanM0 = np.mean(cat1.getvals(vtype="M0"))
cat2 = deepcopy(cat1)
cutils.make_catalog_dc(cat2)
cutils.add_param_noise_bias(cat2, "m_rr", sigma=smrr / 1.5, mean=-mmrr * 3)
# cutils.add_param_noise_bias(cat2, "m_tt", sigma=smtt / 2, mean=0)
# cutils.add_param_noise_bias(cat2, "m_pp", sigma=smpp / 2, mean=0)
cutils.add_param_noise_bias(cat2, "m_rt", sigma=smrt / 3, mean=mmrt)
cutils.add_param_noise_bias(cat2, "m_rp", sigma=smrp / 3, mean=mmrp)

# cutils.add_param_noise(cat2, pdict)
# cutils.add_clvd(cat2, fraction=0.1, positive=True)
# cutils.add_corr_rt_rp(cat2, fraction=0.15)

ranges = {
    "5-12 km": (5, 15, 1),
    "12-20 km": (15, 20, 0.5),
    "20-30 km": (20, 30, 1),
    # "50-70 km": (50, 70, 2),
}

# Setup the new ranges
split_cat1 = cutils.split_cat_mech_depth(cat1, ranges=ranges)
split_cat2 = cutils.split_cat_mech_depth(cat2, ranges=ranges)


cputils.plot_split_gamma_compare(
    split_cat1,
    compare_cat=split_cat2,
    label="CMT3D+",
    compare_label="CMT3D+ (DC+N)",
)


# %%
# Add noise to separate catalogs
cat2 = deepcopy(cat1)
cutils.make_catalog_dc(cat2)


ranges = {
    "5-12 km": (5, 15, 1),
    # "12-20 km": (15, 20, 0.5),
    # "20-30 km": (20, 30, 1),
    # "50-70 km": (50, 70, 2),
}

# Setup the new ranges
split_cat1 = cutils.split_cat_mech_depth(cat1, ranges=ranges)
split_cat2 = cutils.split_cat_mech_depth(cat2, ranges=ranges)


for _mech in ["strike-slip", "normal", "thrust"]:

    if _mech == "normal" or _mech == "thrust":
        fac = -1.0

    else:
        fac = 1

    cutils.add_param_noise_bias(
        split_cat2["5-12 km"]["catalogs"][_mech],
        "m_rr",
        sigma=smrr / 1.2,
        mean=fac * mmrr * 3,
    )
    cutils.add_param_noise_bias(
        split_cat2["5-12 km"]["catalogs"][_mech], "m_rt", sigma=smrt / 2.5, mean=0.0
    )
    cutils.add_param_noise_bias(
        split_cat2["5-12 km"]["catalogs"][_mech], "m_rp", sigma=smrp / 2.5, mean=0.0
    )

print("Mrr", mmrr * 3, smrr / 1.2)
print("Mrt", mmrt, smrt / 2.25)
print("Mrp", mmrp, smrp / 2.25)

# %%
cputils.plot_split_gamma_compare(
    split_cat1,
    compare_cat=split_cat2,
    label="True",
    compare_label="DC+N",
    mechlabel_offset=0.225,
)
fig = plt.gcf()
fig.set_size_inches(8, 2.5)
plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.2)
plt.savefig("split_gamma_compare_test.pdf", dpi=300)
# %%

cc = CompareCatalogs(cat1, cat2, oldlabel="CMT3D+", newlabel="CMT3D+ (DC+N)")
cc.plot_summary(outfile="cmt3d_dc.pdf")


# %%
cputils.plot_parameter_correlations(split_cat1["5-12 km"]["catalogs"]["strike-slip"])

# %%


# gcmt = CMTCatalog.from_file_list(glob("events/gcmt/*"))
# cmt3dp = CMTCatalog.from_file_list(glob("events/gcmt3d_fix/*"))
# catcat = CompareCatalogs(gcmt, cmt3dp, oldlabel="GCMT", newlabel="CMT3D+")

# cputils.plot_parameter_correlations(catcat.old, catcat.new)


# %%

# %%
shallowcat, _ = cat1.filter(maxdict=dict(depth_in_m=12000))
shallowcat_dc, _ = cat2.filter(maxdict=dict(depth_in_m=12000))
# %%

# %%
cputils.plot_parameter_correlations_mech(
    shallowcat,
    parameters=[
        "moment_magnitude",
        "m_rr",
        "m_tt",
        "m_pp",
        "m_rt",
        "m_rp",
        "m_tp",
    ],
    colors=[(0.8, 0.3, 0.3), (0.3, 0.8, 0.3), (0.3, 0.3, 0.8)],
)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig("MT_correlation_shallow.pdf", dpi=300)

# %%

cputils.plot_parameter_correlations_mech(
    shallowcat,
    parameters=[
        "moment_magnitude",
        "lune_kappa",
        "lune_theta",
        "lune_sigma",
        "lune_gamma",
    ],
    colors=[(0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8)],
)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show(block=False)
plt.savefig("Lune_correlation_shallow.pdf", dpi=300)

# %%
cputils.plot_parameter_correlations_mech(
    shallowcat_dc,
    parameters=[
        "moment_magnitude",
        "m_rr",
        "m_tt",
        "m_pp",
        "m_rt",
        "m_rp",
        "m_tp",
        "lune_kappa",
        "lune_theta",
        "lune_sigma",
        "lune_gamma",
        "longitude",
        "latitude",
        "depth_in_m",
        "time_shift",
    ],
    colors=[(0.8, 0.3, 0.3), (0.3, 0.8, 0.3), (0.3, 0.3, 0.8)],
)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
# plt.savefig("correlation_shallow.pdf", dpi=300)


# %%
parameters = [
    "moment_magnitude",
    "m_rr",
    "m_tt",
    "m_pp",
    "m_rt",
    "m_rp",
    "m_tp",
    # "lune_kappa",
    # "lune_theta",
    # "lune_sigma",
    "lune_gamma",
    "longitude",
    "latitude",
    "depth_in_m",
    "time_shift",
]

Acat, limits = cutils.cat2array(
    shallowcat,
    parameters=parameters,
    normalize=True,
)

# %%
Acat_dc, limits = cutils.cat2array(
    shallowcat_dc,
    parameters=parameters,
    normalize=True,
)

# %%


def do_PCA(data):
    ### Representing the Data
    # data has shape (n, d)

    ### Step 1: Standardize the Data along the Features
    standardized_data = (data - data.mean(axis=0)) / data.std(axis=0)

    ### Step 2: Calculate the Covariance Matrix
    # use `ddof = 1` if using sample data (default assumption) and use `ddof = 0` if using population data
    covariance_matrix = np.cov(standardized_data, ddof=1, rowvar=False)

    ### Step 3: Eigendecomposition on the Covariance Matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    ### Step 4: Sort the Principal Components
    # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
    order_of_importance = np.argsort(eigenvalues)[::-1]

    # utilize the sort order to sort eigenvalues and eigenvectors
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:, order_of_importance]  # sort the columns

    ### Step 5: Calculate the Explained Variance
    # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)

    ### Step 6: Reduce the Data via the Principal Components
    k = 2  # select the number of principal components
    reduced_data = np.matmul(
        standardized_data, sorted_eigenvectors[:, :k]
    )  # transform the original data

    ### Step 7: Determine the Explained Variance
    total_explained_variance = sum(explained_variance[:k])

    ### Potential Next Steps: Iterate on the Number of Principal Components
    plt.plot(np.cumsum(explained_variance))
    plt.ylim(0, 1)


do_PCA(Acat)

# %%
mech = shallowcat.getvals(vtype="mtype")
mech_dc = shallowcat_dc.getvals(vtype="mtype")

colordict = {
    "strike-slip": (0.8, 0.2, 0.2),
    "normal": (0.2, 0.8, 0.2),
    "thrust": (0.2, 0.2, 0.8),
    "unknown": (0.5, 0.5, 0.5),
}

colors = np.array([colordict[m] for m in mech])
colors_dc = np.array([colordict[m] for m in mech])

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_rescaled = scaler.fit_transform(Acat)

pca = PCA()
components = pca.fit_transform(data_rescaled)

labels = {
    str(i): f"{var:.1f}" for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

# %%


cputils.crossplot(
    Acat,
    parameters,
    colors,
    {f"{p}": cutils.labeldict[p] for p in parameters},
)

# %%
cputils.crossplot(components, list(labels.keys()), colors, labels)
# plt.show(block=False)

# plt.figure()
# idx1, idx2 = 2, 3
# plt.scatter(Acat[:, idx1], Acat[:, idx2], c=colors, s=10, alpha=0.5, edgecolors="none")
# plt.xlabel(parameters[idx1])
# plt.xlabel(parameters[idx2])
# plt.show(block=False)


# %%

for _mech in ["strike-slip", "normal", "thrust", "unknown"]:

    idx = np.where(mech_dc == _mech)[0]

    cputils.crossplot(
        Acat_dc[idx, :],
        parameters,
        [colors[i] for i in idx],
        {f"{p}": cutils.labeldict[p] for p in parameters},
    )
    plt.suptitle("Mechanism: " + cutils.labeldict[_mech])
    plt.savefig(f"mechanism_{_mech}_dc.pdf", dpi=300)
    plt.close("all")

    scaler = StandardScaler()
    data_rescaled = scaler.fit_transform(Acat_dc[idx, :])

    pca = PCA()
    components = pca.fit_transform(data_rescaled)

    labels = {
        str(i): f"{var:.1f}"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    cputils.crossplot(
        components,
        list(labels.keys()),
        [colors[i] for i in idx],
        labels,
    )

    plt.suptitle("Mechanism: " + cutils.labeldict[_mech] + " PCA")
    plt.savefig(f"mechanism_{_mech}_dc_PCA.pdf", dpi=300)
    plt.close("all")
