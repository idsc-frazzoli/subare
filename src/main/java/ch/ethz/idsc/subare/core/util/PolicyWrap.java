// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Random;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.alg.Accumulate;
import ch.ethz.idsc.tensor.alg.Last;
import ch.ethz.idsc.tensor.sca.Chop;

/** class picks action based on distribution defined by given {@link Policy} */
public class PolicyWrap {
  private final Policy policy;
  private final Random random;

  public PolicyWrap(Policy policy, Random random) {
    this.policy = policy;
    this.random = random;
  }

  public Tensor next(Tensor state, Tensor actions) {
    Tensor prob = Accumulate.of( //
        Tensor.of(actions.flatten(0).map(action -> policy.probability(state, action))));
    // ---
    if (!Chop.isZeros(Last.of(prob).subtract(RealScalar.ONE)))
      throw TensorRuntimeException.of(prob);
    // ---
    // TODO use random variate, wherever java.util.Random is used in project
    Scalar threshold = DoubleScalar.of(random.nextDouble());
    int index = 0;
    for (; index < prob.length(); ++index)
      if (Scalars.lessThan(threshold, prob.Get(index)))
        break;
    return actions.get(index);
  }
}
