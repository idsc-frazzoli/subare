// code by jph
package ch.ethz.idsc.subare.ch02.streets;

import java.util.Arrays;
import java.util.List;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** example network by Julian
 * three possible actions */
class BridgeNetwork extends Network {
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

  Tensor bias = Tensors.vectorInt(-1, 0, 0, -1, 0);

  @Override
  Scalar streetBias(int index) {
    return bias.Get(index);
  }
}
