// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;

public class CarRental implements StandardModel {
  final Tensor states = Flatten.of(Array.of(Tensors::vector, 21, 21), 1).unmodifiable();

  public CarRental() {
    // TODO Auto-generated constructor stub
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Scalar qsa(Tensor state, Tensor action, Tensor gvalues) {
    // TODO Auto-generated method stub
    return null;
  }
}
