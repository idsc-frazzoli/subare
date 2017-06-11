// code by jph
package ch.ethz.idsc.subare.core.alg;

import ch.ethz.idsc.subare.core.ActionValueInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

enum StaticHelper {
  ;
  // ---
  // test that probabilities add up to 1
  static void assertConsistent(Tensor keys, ActionValueInterface actionValueInterface) {
    keys.flatten(0).parallel() //
        .forEach(pair -> _isConsistent(actionValueInterface, pair.get(0), pair.get(1)));
  }

  private static void _isConsistent(ActionValueInterface actionValueInterface, Tensor state, Tensor action) {
    Scalar norm = actionValueInterface.transitions(state, action).flatten(0) //
        .map(next -> actionValueInterface.transitionProbability(state, action, next)) //
        .reduce(Scalar::add).get();
    if (!norm.equals(RealScalar.ONE))
      throw new RuntimeException(); // probabilities have to sum up to 1
  }
}
