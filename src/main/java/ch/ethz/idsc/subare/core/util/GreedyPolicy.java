// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.RealScalar;

/** TODO JAN put comment somewhere:
 * 
 * exact implementation of equiprobable greedy policy:
 * if two or more states s1,s2, ... have equal value
 * v(s1)==v(s2)
 * then they are assigned equal probability
 * 
 * in case there is no unique maximum value
 * there are infinitely many greedy policies
 * and not a unique one policy.
 * 
 * @param standardModel
 * @param vs of standardModel.states()
 * @return */
public enum GreedyPolicy {
  ;
  public static Policy of(DiscreteModel discreteModel, QsaInterface qsa) {
    return new EGreedyPolicy(discreteModel, qsa, RealScalar.ZERO);
  }

  public static Policy of(StandardModel standardModel, VsInterface vs) {
    return new EGreedyPolicy(standardModel, vs, RealScalar.ZERO, standardModel.states());
  }
}
