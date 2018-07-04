package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Norm;
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.sca.Power;

public enum MonteCarloErrorAnalysis {
  LINEAR_QSA() {
    @Override
    public Scalar getError(MonteCarloInterface monteCarloInterface, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      return DiscreteValueFunctions.distance(refQsa, currentQsa).Get();
    }
  },
  SQUARE_QSA() {
    @Override
    public Scalar getError(MonteCarloInterface monteCarloInterface, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      return Power.of(DiscreteValueFunctions.distance(refQsa, currentQsa, Norm._2).Get(), 2);
    }
  },
  LINEAR_POLICY() {
    @Override
    public Scalar getError(MonteCarloInterface monteCarloInterface, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      return Total.of(Loss.asQsa(monteCarloInterface, refQsa, currentQsa).values()).Get();
    }
  },
  SQUARE_POLICY() {
    @Override
    public Scalar getError(MonteCarloInterface monteCarloInterface, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      Tensor errors = Loss.asQsa(monteCarloInterface, refQsa, currentQsa).values();
      return Total.of(errors.pmul(errors)).Get();
    }
  },;
  public abstract Scalar getError(MonteCarloInterface monteCarloInterface, DiscreteQsa refQsa, DiscreteQsa currentQsa);
}
