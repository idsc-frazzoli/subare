// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.gambler;

import java.io.File;
import java.io.IOException;

import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.Put;

class QL_Gambler {
  public static void main(String[] args) throws IOException {
    Gambler gambler = new Gambler(100, RationalScalar.of(4, 10));
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    QLearning qLearning = new QLearning( //
        gambler, new EquiprobablePolicy(gambler), //
        gambler, //
        qsa, RealScalar.ONE, RealScalar.of(.1)); // TODO ask jz
    qLearning.simulate(30000);
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    DiscreteVs discreteVs = DiscreteVs.create(gambler, qsa);
    // discreteVs.print();
    Tensor result = discreteVs.values();
    Put.of(new File("/home/datahaki/ql_gambler"), result);
  }
}
