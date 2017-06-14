// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import ch.ethz.idsc.subare.core.adapter.DeterministicStandardModel;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

class Dynamaze extends DeterministicStandardModel {
  @Override
  public Tensor states() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Tensor actions(Tensor state) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Scalar gamma() {
    // TODO Auto-generated method stub
    return null;
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    // TODO Auto-generated method stub
    return null;
  }
}
