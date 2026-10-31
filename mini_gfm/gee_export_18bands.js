/*
 CarbonAce Mini-GFM source: 3 years x 6 bands = 18 band COG.
 Use Earth Engine Code Editor https://code.earthengine.google.com
*/
var aoi = ee.Geometry.Rectangle([126.88, 37.53, 126.97, 37.60]);
var years = [2023, 2024, 2025];
var seasonStartMonth = 4;
var seasonEndExclusiveMonth = 11; // Apr 1 - Oct 31 inclusive
var exportCrs = 'EPSG:32652';
var exportScale = 20;

Map.centerObject(aoi, 12);
Map.addLayer(aoi, {color: 'red'}, 'AOI', false);

function maskClouds(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(0).and(scl.neq(1))
    .and(scl.neq(3)).and(scl.neq(8))
    .and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return img.updateMask(mask)
    .select(['B3', 'B4', 'B8', 'B11'])
    .multiply(0.0001);
}

var demcol = ee.ImageCollection('COPERNICUS/DEM/GLO30').filterBounds(aoi);
var demproj = ee.Image(demcol.first()).select('DEM').projection();
var elev = demcol.select('DEM').mosaic()
  .setDefaultProjection(demproj).rename('Elevation');
var slope = ee.Terrain.slope(elev).rename('Slope');

function makeYear(year) {
  var start = ee.Date.fromYMD(year, seasonStartMonth, 1);
  var stop = ee.Date.fromYMD(year, seasonEndExclusiveMonth, 1);
  var scenes = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi).filterDate(start, stop)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)).map(maskClouds);
  print('Sentinel-2 images ' + year, scenes.size());
  var composite = scenes.median();
  return ee.Image.cat([
    composite.normalizedDifference(['B8', 'B4']).rename('NDVI_' + year),
    composite.normalizedDifference(['B3', 'B8']).rename('NDWI_' + year),
    composite.normalizedDifference(['B11', 'B8']).rename('NDBI_' + year),
    composite.normalizedDifference(['B8', 'B11']).rename('NDMI_' + year),
    slope.rename('Slope_' + year),
    elev.rename('Elevation_' + year)
  ]).toFloat().clip(aoi);
}

var stack = ee.Image.cat(years.map(makeYear));
print('Input to Mini-GFM:', stack.bandNames());
print('Total bands:', stack.bandNames().size());
Map.addLayer(stack.select('NDVI_2025'),
  {min:-0.1,max:0.8,palette:['brown','yellow','green']}, 'NDVI 2025');

Export.image.toDrive({
  image: stack.unmask(-9999).toFloat(),
  description: 'CarbonAce_MiniGFM_18bands',
  folder: 'GEE_Export',
  fileNamePrefix: 'GeoAI_2023_2025_18Bands',
  region: aoi, scale: exportScale, crs: exportCrs,
  maxPixels: 1e13, fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true, noData: -9999}
});
