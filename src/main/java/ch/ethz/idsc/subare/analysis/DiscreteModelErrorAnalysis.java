// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.red.Norm;
import ch.ethz.idsc.tensor.red.Norm2Squared;
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.sca.Power;

public enum DiscreteModelErrorAnalysis {
  LINEAR_QSA() {
    @Override
    public Scalar getError(DiscreteModel discreteModel, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      return DiscreteValueFunctions.distance(refQsa, currentQsa);
    }
  },
  SQUARE_QSA() {
    @Override
    public Scalar getError(DiscreteModel discreteModel, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      return Power.of(DiscreteValueFunctions.distance(refQsa, currentQsa, Norm._2), 2);
    }
  },
  LINEAR_POLICY() {
    @Override
    public Scalar getError(DiscreteModel discreteModel, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      return (Scalar) Total.of(Loss.asQsa(discreteModel, refQsa, currentQsa).values());
    }
  },
  SQUARE_POLICY() {
    @Override
    public Scalar getError(DiscreteModel discreteModel, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      return Norm2Squared.ofVector(Loss.asQsa(discreteModel, refQsa, currentQsa).values());
    }
  };
  /** @param discreteModel
   * @param refQsa
   * @param currentQsa
   * @return */
  public abstract Scalar getError(DiscreteModel discreteModel, DiscreteQsa refQsa, DiscreteQsa currentQsa);
}
