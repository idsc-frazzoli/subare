// code by jph
package ch.ethz.idsc.subare.plot;

import java.util.ArrayList;
import java.util.List;

public class LabelList {
  private final List<ComparableLabel> list = new ArrayList<>();

  public ComparableLabel get(int index) {
    while (list.size() <= index)
      list.add(new ComparableLabel(list.size()));
    return list.get(index);
  }

  public ComparableLabel next() {
    return get(list.size());
  }
}
