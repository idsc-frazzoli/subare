// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Random;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.alg.Accumulate;
import ch.ethz.idsc.tensor.alg.Last;
import ch.ethz.idsc.tensor.sca.Chop;

/** class picks action based on distribution defined by given {@link PolicyInterface} */
public class PolicyWrap {
  private final PolicyInterface policyInterface;
  private final Random random;

  public PolicyWrap(PolicyInterface policyInterface, Random random) {
    this.policyInterface = policyInterface;
    this.random = random;
  }

  public Tensor next(Tensor state, Tensor actions) {
    Tensor prob = Accumulate.of( //
        Tensor.of(actions.flatten(0).map(action -> policyInterface.policy(state, action))));
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
