# %%
import os
from cmt3d.cmt_catalog import CMTCatalog
from cmt3d.source import CMTSource

catfile = "/Users/lucassawade/PDrive/Research/Projects/GCMT3D/children/ReykjanesRidge/gcmtevents.txt"


def split_cat(catfile):
    with open(catfile, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith(" "):
            start = i
        if line.startswith("Mtp") and i > 0:
            end = i + 1
            yield "".join(lines[start:end])


def split_eventfile(catfile, outdir="./"):

    # Check if dir exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Write catalog
    filelist = []
    for filecontent in split_cat(catfile):
        eventid = filecontent.split("\n")[1].split(":")[1].strip()
        file = f"{outdir}/{eventid}.txt"
        filelist.append(file)
        with open(file, "w") as f:
            f.write(filecontent)

    return filelist


filelist = split_eventfile(
    catfile,
    outdir="/Users/lucassawade/PDrive/Research/Projects/GCMT3D/children/ReykjanesRidge/events",
)

catalog = CMTCatalog.from_file_list(filelist)
# %%
from cmt3d.viz.catalog_plot_utils import plot_gmt_cat

region = [-36.0, -34.4, 53.6, 54.4]
plot_gmt_cat(
    catalog, region=region, topography=True, resolution="01s", outfile="reykjanes.pdf"
)

# %%
# Create GF location file

locations_file = "/Users/lucassawade/PDrive/Research/Projects/GCMT3D/children/ReykjanesRidge/GF_LOCATIONS"


# Open GF locations file for each compenent
with open(locations_file, "w") as f:

    # Loop over provided target locations
    for event in catalog:
        _lat = event.latitude
        _lon = event.longitude
        _dep = event.depth_in_m / 1000.0
        f.write(f"{_lat:9.4f}   {_lon:9.4f}   {_dep:9.4f}\n")
