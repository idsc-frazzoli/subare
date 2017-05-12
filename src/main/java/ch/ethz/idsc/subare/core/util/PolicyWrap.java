// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Random;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Accumulate;
import ch.ethz.idsc.tensor.alg.Last;
import ch.ethz.idsc.tensor.sca.Chop;

public class PolicyWrap {
  private final Random random = new Random();
  private final PolicyInterface policyInterface;

  public PolicyWrap(PolicyInterface policyInterface) {
    this.policyInterface = policyInterface;
  }

  public Tensor next(Tensor state, Tensor actions) {
    Tensor prob = Accumulate.of( //
        Tensor.of(actions.flatten(0).map(action -> policyInterface.policy(state, action))));
    if (!Chop.isZeros(Last.of(prob).subtract(RealScalar.ONE)))
      throw new RuntimeException("no distribution " + prob);
    Scalar threshold = DoubleScalar.of(random.nextDouble());
    int index = 0;
    for (; index < prob.length(); ++index)
      if (Scalars.lessThan(threshold, prob.Get(index)))
        break;
    return actions.get(index);
  }
}
