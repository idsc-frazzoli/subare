// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.ActionValueInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** applicable for models that have deterministic move and reward */
public abstract class NonDeterministicStandardModel implements StandardModel, ActionValueInterface {
  @Override
  public final Scalar qsa(Tensor state, Tensor action, VsInterface gvalues) {
    Scalar sum = transitions(state, action).flatten(0) //
        .map(next -> transitionProbability(state, action, next).multiply(gvalues.value(next))) //
        .reduce(Scalar::add).get();
    return expectedReward(state, action).add(sum);
  }
}
