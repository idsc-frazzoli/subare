// code by jph
package ch.ethz.idsc.subare.demo.streets;

import java.util.Arrays;
import java.util.List;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** example network by Julian
 * three possible actions */
class BridgeNetwork extends Network {
  private Tensor affine = Tensors.vectorInt(-3, 0, 0, -3, 0).unmodifiable();
  private Tensor linear = Tensors.vectorInt(0, -1, -1, 0, 0).unmodifiable();

  @Override
  int actions() {
    return 3;
  }

  @Override
  int streets() {
    return 5;
  }

  @Override
  List<Integer> streetsFromAction(int k) {
    switch (k) {
    case 0:
      return Arrays.asList(0, 1);
    case 1:
      return Arrays.asList(2, 3);
    case 2:
      return Arrays.asList(2, 4, 1);
    }
    return null;
  }

  @Override
  Tensor affine() {
    return affine;
  }

  @Override
  Tensor linear() {
    return linear;
  }
}
