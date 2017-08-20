// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.alg.IterativePolicyEvaluation;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

// TODO check again
enum IPE_InfiniteVariance {
  ;
  public static void main(String[] args) {
    StandardModel standardModel = new InfiniteVariance();
    Policy policy = new ConstantPolicy(RationalScalar.of(9, 10));
    IterativePolicyEvaluation a = new IterativePolicyEvaluation( //
        standardModel, policy);
    a.until(RealScalar.of(.0001));
    System.out.println(a.iterations());
    a.vs().print(Round._2);
  }
}
