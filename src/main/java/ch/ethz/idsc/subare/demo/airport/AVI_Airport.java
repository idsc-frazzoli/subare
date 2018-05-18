package ch.ethz.idsc.subare.demo.airport;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Tensors;

enum AVI_Airport {
  ;
  public static void main(String[] args) {
    Airport airport = new Airport();

    DiscreteQsa qsa = ActionValueIterations.solve(airport, DecimalScalar.of(.0001));
    
    DiscreteUtils.print(qsa);
    qsa.value(Tensors.vector(0,5,0), Tensors.vector(1,-1,1));
    Policy policy = GreedyPolicy.bestEquiprobable(airport, qsa);
    
    Policies.print(policy, airport.states());
    
  }
}
