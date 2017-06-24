// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Random;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.EmpiricalDistribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.red.Total;
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
    Tensor pdf = Tensor.of(actions.flatten(0).map(action -> policy.probability(state, action)));
    if (!Chop.isZeros(Total.of(pdf).subtract(RealScalar.ONE)))
      throw TensorRuntimeException.of(pdf);
    Distribution distribution = EmpiricalDistribution.fromUnscaledPDF(pdf);
    return actions.get(RandomVariate.of(distribution, random).number().intValue());
  }
}
