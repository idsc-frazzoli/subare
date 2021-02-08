// code by jph
package ch.ethz.idsc.subare.core.alg;

import ch.ethz.idsc.subare.core.ActionValueInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;

/* package */ enum StaticHelper {
  ;
  // test that probabilities add up to 1
  static void assertConsistent(Tensor keys, ActionValueInterface actionValueInterface) {
    keys.stream().parallel() //
        .forEach(pair -> _isConsistent(actionValueInterface, pair.get(0), pair.get(1)));
  }

  private static void _isConsistent(ActionValueInterface actionValueInterface, Tensor state, Tensor action) {
    Scalar norm = actionValueInterface.transitions(state, action).stream() //
        .map(next -> actionValueInterface.transitionProbability(state, action, next)) //
        .reduce(Scalar::add).get();
    if (!norm.equals(RealScalar.ONE)) {
      System.out.println("state =" + state);
      System.out.println("action=" + action);
      actionValueInterface.transitions(state, action).stream().forEach(next -> {
        Scalar prob = actionValueInterface.transitionProbability(state, action, next);
        System.out.println(next + " " + prob);
      });
      System.exit(0);
      throw TensorRuntimeException.of(norm, state, action); // probabilities have to sum up to 1
    }
  }
}
