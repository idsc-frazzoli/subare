// code by jph
package ch.ethz.idsc.subare.util;

import java.util.List;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Dimensions;

public class ImageResize {
  public static Tensor of(Tensor image, int factor) {
    List<Integer> list = Dimensions.of(image);
    return Tensors.matrix((i, j) -> image.get(i / factor, j / factor), //
        list.get(0) * factor, //
        list.get(1) * factor);
  }
}
