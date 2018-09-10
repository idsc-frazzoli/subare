// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Sign;
import ch.ethz.idsc.tensor.sca.Sqrt;

public enum UcbUtils {
  ;
  public static DiscreteQsa getUcbInQsa(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
    DiscreteQsa qsaWithUcb = DiscreteQsa.build(discreteModel);
    for (Tensor state : discreteModel.states()) {
      for (Tensor action : discreteModel.actions(state)) {
        qsaWithUcb.assign(state, action, getUpperConfidenceBound(state, action, qsa, sac, discreteModel));
      }
    }
    return qsaWithUcb;
  }

  public static Scalar getUpperConfidenceBound(Tensor state, Tensor action, QsaInterface qsa, StateActionCounter sac, DiscreteModel discreteModel) {
    Scalar qsaValue = qsa.value(state, action);
    Tensor key = StateAction.key(state, action);
    Scalar Nta = sac.stateActionCount(key);
    if (Scalars.isZero(Nta))
      return DoubleScalar.POSITIVE_INFINITY;
    Scalar actionSize = RealScalar.of(discreteModel.actions(state).length());
    Scalar bias = Sqrt.of(sac.stateCount(state)).divide(Nta);
    Scalar sign = Sign.isPositive(qsaValue) ? RealScalar.ONE : RealScalar.of(-1);
    return qsaValue.multiply((RealScalar.ONE.add(bias.multiply(sign))));
  }
}
