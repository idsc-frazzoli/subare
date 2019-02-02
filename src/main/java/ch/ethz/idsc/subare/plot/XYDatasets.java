// code by fluric, amodeus
package ch.ethz.idsc.subare.plot;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import ch.ethz.idsc.tensor.Tensor;

public enum XYDatasets {
  ;
  public static XYDataset create(Map<String, Tensor> map) {
    return create(map, 0);
  }

  public static XYDataset create(Map<String, Tensor> map, int index) {
    XYSeriesCollection xySeriesCollection = new XYSeriesCollection();
    for (Entry<String, Tensor> entry : map.entrySet()) {
      XYSeries xySeries = new XYSeries(entry.getKey());
      Tensor xyData = entry.getValue();
      for (int count = 0; count < xyData.length(); ++count)
        xySeries.add(xyData.get(count).Get(0).number(), xyData.get(count).Get(1 + index).number());
      xySeriesCollection.addSeries(xySeries);
    }
    return xySeriesCollection;
  }

  public static XYDataset create(List<Tensor> XYs, List<String> names) {
    final XYSeriesCollection collection = new XYSeriesCollection();
    // create dataset
    for (int i = 0; i < names.size(); ++i) {
      XYSeries series = new XYSeries(names.get(i));
      Tensor XY = XYs.get(i);
      for (int j = 0; j < XY.length(); ++j)
        series.add(XY.get(j).Get(0).number(), XY.get(j).Get(1).number());
      collection.addSeries(series);
    }
    return collection;
  }
}
