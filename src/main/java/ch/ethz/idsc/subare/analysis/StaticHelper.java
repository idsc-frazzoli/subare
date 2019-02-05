// code by jph
package ch.ethz.idsc.subare.analysis;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.jfree.data.xy.XYSeries;

import ch.ethz.idsc.subare.plot.XYSeriesList;
import ch.ethz.idsc.tensor.Tensor;

/* package */ enum StaticHelper {
  ;
  public static XYSeriesList create(String string, Tensor xy) {
    return create(Collections.singletonMap(string, xy));
  }

  public static XYSeriesList create(Map<String, Tensor> map) {
    return create(map, 0);
  }

  public static XYSeriesList create(Map<String, Tensor> map, int index) {
    XYSeriesList xySeriesList = new XYSeriesList();
    for (Entry<String, Tensor> entry : map.entrySet()) {
      XYSeries xySeries = xySeriesList.add(entry.getValue());
      xySeries.setKey(entry.getKey());
    }
    return xySeriesList;
  }

  public static XYSeriesList create(List<Tensor> list, List<String> names) {
    if (list.size() != names.size())
      throw new RuntimeException();
    XYSeriesList xySeriesList = new XYSeriesList();
    for (int i = 0; i < names.size(); ++i) {
      XYSeries xySeries = xySeriesList.add(list.get(i));
      xySeries.setKey(names.get(i));
    }
    return xySeriesList;
  }
}
