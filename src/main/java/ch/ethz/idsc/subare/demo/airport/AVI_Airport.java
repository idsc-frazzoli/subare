package ch.ethz.idsc.subare.demo.airport;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;

enum AVI_Airport {
  ;
  public static void main(String[] args) throws Exception {
    Airport airport = new Airport();
    DiscreteQsa qsa = ActionValueIterations.solve(airport, DecimalScalar.of(.0001));
    DiscreteUtils.print(qsa);
    Policy policyQsa = GreedyPolicy.bestEquiprobable(airport, qsa);
    Policies.print(policyQsa, airport.states());
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(airport);
    int batches = 100;
    for (int index = 0; index < batches; ++index) {
      Policy policyMC = EGreedyPolicy.bestEquiprobable(airport, mces.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(airport, policyMC, mces);
    }
    Policies.print(EGreedyPolicy.bestEquiprobable(airport, mces.qsa(), RealScalar.of(.1)), airport.states());
  }
}
