// code by jph
package ch.ethz.idsc.subare.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.tensor.Tensor;

public class Index {
  public static Index build(Tensor tensor) {
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

  public Tensor keys() {
    return tensor.unmodifiable();
  }

  public Tensor get(int index) {
    return tensor.get(index).unmodifiable();
  }

  public boolean containsKey(Tensor action) {
    return map.containsKey(action);
  }

  public int of(Tensor row) {
    if (!containsKey(row))
      System.out.println("unknown key=" + row);
    return map.get(row);
  }

  public int size() {
    return tensor.length();
  }
}
