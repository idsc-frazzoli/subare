// code by fluric, amodeus
package ch.ethz.idsc.subare.plot;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import ch.ethz.idsc.tensor.Tensor;

public enum XYDatasets {
  ;
  public static XYDataset create(String string, Tensor xy) {
    return create(Collections.singletonMap(string, xy));
  }

  public static XYDataset create(Map<String, Tensor> map) {
    return create(map, 0);
  }

  public static XYDataset create(Map<String, Tensor> map, int index) {
    XYSeriesCollection xySeriesCollection = new XYSeriesCollection();
    for (Entry<String, Tensor> entry : map.entrySet()) {
      XYSeries xySeries = new XYSeries(entry.getKey());
      for (Tensor row : entry.getValue())
        xySeries.add(row.Get(0).number(), row.Get(1 + index).number());
      xySeriesCollection.addSeries(xySeries);
    }
    return xySeriesCollection;
  }

  public static XYDataset create(List<Tensor> list, List<String> names) {
    if (list.size() != names.size())
      throw new RuntimeException();
    XYSeriesCollection xySeriesCollection = new XYSeriesCollection();
    for (int i = 0; i < names.size(); ++i) {
      XYSeries xySeries = new XYSeries(names.get(i));
      Tensor tensor = list.get(i);
      for (Tensor row : tensor)
        xySeries.add(row.Get(0).number(), row.Get(1).number());
      xySeriesCollection.addSeries(xySeries);
    }
    return xySeriesCollection;
  }
}
