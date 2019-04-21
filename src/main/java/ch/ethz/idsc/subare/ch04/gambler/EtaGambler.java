// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.Arrays;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.OnPolicyStateDistribution;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.ArrayPad;
import ch.ethz.idsc.tensor.alg.Normalize;
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.red.VectorTotal;
import ch.ethz.idsc.tensor.sca.Increment;
import ch.ethz.idsc.tensor.sca.Sign;

/* package */ enum EtaGambler {
  ;
  public static void main(String[] args) {
    Gambler gambler = new Gambler(10, RationalScalar.of(4, 10));
    // Policy policy = EquiprobablePolicy.create(gambler);
    Policy policy = GamblerHelper.getOptimalPolicy(gambler);
    OnPolicyStateDistribution opsd = new OnPolicyStateDistribution(gambler, gambler, policy);
    Tensor values = //
        ArrayPad.of(Array.zeros(9).map(Increment.ONE), Arrays.asList(1), Arrays.asList(1));
    values.map(Sign::requirePositiveOrZero);
    values = Normalize.with(VectorTotal.FUNCTION).apply(values);
    Scalar scalar = VectorTotal.FUNCTION.apply(values);
    System.out.println("sum=" + scalar);
    DiscreteVs vs = DiscreteVs.build(gambler.states(), values);
    for (int count = 0; count < 10; ++count) {
      vs = opsd.iterate(vs);
      System.out.println(vs.values());
      System.out.println("total=" + Total.of(vs.values()));
    }
  }
}
