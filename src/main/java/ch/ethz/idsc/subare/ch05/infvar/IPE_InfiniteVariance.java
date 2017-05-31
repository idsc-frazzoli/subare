// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.alg.IterativePolicyEvaluation;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

class IPE_InfiniteVariance {
  public static void main(String[] args) {
    StandardModel standardModel = new InfiniteVariance();
    PolicyInterface policyInterface = new ConstantPolicy(RationalScalar.of(9, 10));
    IterativePolicyEvaluation a = new IterativePolicyEvaluation( //
        standardModel, policyInterface);
    a.until(RealScalar.of(.0001));
    System.out.println(a.iterations());
    a.vs().print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
