// code by jph
package ch.ethz.idsc.subare.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.sca.Exp;

public class PoissonDistribution {
  private static final int PRECOMPUTE_LENGTH = 16;
  private static final Map<Scalar, PoissonDistribution> MEMO = new HashMap<>();

  public static PoissonDistribution of(Scalar lambda) {
    if (!MEMO.containsKey(lambda))
      MEMO.put(lambda, new PoissonDistribution(lambda));
    return MEMO.get(lambda);
  }

  final Scalar lambda;
  private final Tensor values = Tensors.empty();

  private PoissonDistribution(Scalar lambda) {
    this.lambda = lambda;
    values.append(Exp.of(lambda.negate()));
    apply(PRECOMPUTE_LENGTH);
  }

  public Scalar apply(int n) {
    while (values.length() <= n) {
      int index = values.length();
      Scalar factor = lambda.divide(RealScalar.of(index));
      values.append(values.Get(index - 1).multiply(factor));
    }
    return values.Get(n);
  }
}
