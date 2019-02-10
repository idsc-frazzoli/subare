// code by jph
package ch.ethz.idsc.subare.analysis;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ch.ethz.idsc.subare.util.plot.VisualRow;
import ch.ethz.idsc.subare.util.plot.VisualSet;
import ch.ethz.idsc.tensor.Tensor;

/* package */ enum StaticHelper {
  ;
  public static VisualSet create(String string, Tensor xy) {
    return create(Collections.singletonMap(string, xy));
  }

  public static VisualSet create(Map<String, Tensor> map) {
    return create(map, 0);
  }

  public static VisualSet create(Map<String, Tensor> map, int index) {
    VisualSet visualSet = new VisualSet();
    for (Entry<String, Tensor> entry : map.entrySet()) {
      VisualRow visualRow = visualSet.add(entry.getValue());
      visualRow.setLabel(entry.getKey());
    }
    return visualSet;
  }

  public static VisualSet create(List<Tensor> list, List<String> names) {
    if (list.size() != names.size())
      throw new RuntimeException();
    VisualSet visualSet = new VisualSet();
    for (int i = 0; i < names.size(); ++i) {
      VisualRow visualRow = visualSet.add(list.get(i));
      visualRow.setLabel(names.get(i));
    }
    return visualSet;
  }
}
