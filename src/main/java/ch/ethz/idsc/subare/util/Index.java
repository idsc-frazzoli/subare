// code by jph
package ch.ethz.idsc.subare.util;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;

/** index is similar to a database index over the 0-level entries of the tensor.
 * the index allows fast checks for containment and gives the position of the key
 * in the original tensor of keys */
public class Index implements Serializable {
  public static Index build(Tensor tensor) {
    return new Index(tensor);
  }

  // ---
  private final Tensor keys;
  private final Map<Tensor, Integer> map = new HashMap<>();

  private Index(Tensor keys) {
    this.keys = keys;
    int index = -1;
    for (Tensor key : keys)
      map.put(key, ++index);
  }

  public Tensor keys() {
    return keys;
  }

  public Tensor get(int index) {
    return keys.get(index).unmodifiable();
  }

  public boolean containsKey(Tensor key) {
    return map.containsKey(key);
  }

  public int of(Tensor row) {
    if (containsKey(row))
      return map.get(row);
    throw TensorRuntimeException.of(row);
  }

  public int size() {
    return keys.length();
  }
}
