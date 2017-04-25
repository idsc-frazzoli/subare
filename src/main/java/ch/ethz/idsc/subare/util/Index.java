// code by jph
package ch.ethz.idsc.subare.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.tensor.Tensor;

public class Index {
  public static Index of(Tensor tensor) {
    return new Index(tensor);
  }

  private final Tensor tensor;
  private final Map<Tensor, Integer> map = new HashMap<>();

  private Index(Tensor tensor) {
    this.tensor = tensor.copy();
    int index = -1;
    for (Tensor row : this.tensor)
      map.put(row, ++index);
  }

  public Tensor get(int index) {
    return tensor.get(index);
  }

  public int indexOf(Tensor row) {
    return map.get(row);
  }

  public int size() {
    return tensor.length();
  }
}
