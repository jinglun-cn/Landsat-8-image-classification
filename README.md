# Landsat-8 Classification

---------------------

### Prerequisites

Python >=3.5

Tensorflow >=2.0

[GDAL](https://gdal.org/)

[Rasterio](https://github.com/mapbox/rasterio/)

[Shapely](https://github.com/Toblerity/Shapely/)

### Sites for Training

The sites of interests consist of 529 locations in 32 states in the US, which are provided in the following file:

```bash
./sites_train.csv
```

![sites](https://github.com/jinglun-cn/landsat-classification/blob/master/figure/sites.jpg?raw=true)

### Download and Process Data

```bash
python data_processing.py
```

The script downloads [Landsat-8](https://www.usgs.gov/land-resources/nli/landsat/landsat-8?qt-science_support_page_related_con=0#qt-science_support_page_related_con) and [National Land Cover Database 2016(NLCD2016)](https://www.mrlc.gov/national-land-cover-database-nlcd-2016) from Amazon S3.

For Landsat-8, input Lat/Lon are converted to Path/Row to match
the product ID of scenes. The level 1 scenes in 2016 with corresponding path/row and the least cloud coverage are downloaded from Amazon S3 Storage. OLI bands 1-7 and 9 are extracted.

The whole process might take a long time and requires at least 22G disk space, depending on the exact input sites.

For each input site, the script crops a 3840 m x 3840 m rectangle images, each with the given site located in the center.

The extracted data are saved as a numpy array with datatype "uint16" and shape (sample_index, x-coor, y-coor, band). 


### Extracted Data Example


Below is an example. First 8 images are from Landsat-8. The last image is from NLCD.

![example](https://github.com/jinglun-cn/landsat-classification/blob/master/figure/example.png?raw=true)

### Train the CNN

```bash
python train.py
```

The script reads numpy array, generates small patches and train the CNN.

In [[Sharma et al., 2017]](https://www.sciencedirect.com/science/article/pii/S0893608017301806), a patch-based CNN is trained and tested using Landsat-8 data within the Florida Everglades ecosystem. As our data are significantly larger and more diverse, the oringinal model achieved only 78% in validation accuracy. Our model is able to to improve the validation accuracy to 89%.

### Classify sites using pretrained model

```bash
python classify.py
```

The script loads the pretrained model from:

```bash
./pretrained.hdf5
```

The sites for classification are given in:

```bash
./sites_classify.csv
```

Landsat-8 data are downloaded, cropped and classified.

The class labels are stored in the output array:

```bash
./arr_cls.npy
```

As the model is patch-based, the boundary pixels are unclassified. One can preset the size of image to classify a larger area.


### * Process LiDAR point cloud (LPC) data to estimate elevation
Input: [LiDAR point cloud (LPC)](https://viewer.nationalmap.gov/basic/) data (.las file) and extracted Landsat8-NLCD data (.tif file).

For each pixel in the extracted raster file, the program estimates
the average elevation by calculating the mean of a 10*10 square
of LPC data. The elevation is classified into three categories(low, medium and high).

The average elevation and elevation classes are outputs. Both of
them are 128*128 arrays.

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
