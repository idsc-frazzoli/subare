// code by jph
package ch.ethz.idsc.subare.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Last;
import ch.ethz.idsc.tensor.sca.Exp;

/** TODO cite online reference */
public class PoissonDistribution {
  /* package */ static final int PRECOMPUTE_LENGTH = 16;
  /* package */ static final Map<Scalar, PoissonDistribution> MEMO = new HashMap<>();

  public static PoissonDistribution of(Scalar lambda) {
    if (!MEMO.containsKey(lambda))
      MEMO.put(lambda, new PoissonDistribution(lambda));
    return MEMO.get(lambda);
  }

  private final Scalar lambda;
  private final Tensor values = Tensors.empty();

  private PoissonDistribution(Scalar lambda) {
    this.lambda = lambda;
    values.append(Exp.of(lambda.negate()));
    apply(PRECOMPUTE_LENGTH - 1);
  }

  public Scalar lambda() {
    return lambda;
  }

  public Scalar apply(int n) {
    if (n < 0)
      throw new RuntimeException("negative: " + n);
    while (values.length() <= n) {
      Scalar factor = lambda.divide(RealScalar.of(values.length()));
      values.append(Last.of(values).multiply(factor));
    }
    return values.Get(n);
  }

  public Tensor values(int length) {
    return Tensors.vector(this::apply, length);
  }

  /* package */ Tensor values() {
    return values;
  }
}
