// code by jph
package ch.ethz.idsc.subare.ch02.prison;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.tensor.io.Put;

class Export {
  public static void main(String[] args) throws IOException {
    // List<Supplier<Agent>> list = AgentSupplier.getOptimists(.01, .8, 30);
    // List<Supplier<Agent>> list = AgentSupplier.getUCBs(0, 6, 30);
    List<Supplier<Agent>> list = AgentSupplier.getEgreedyC(0.1, .8, 20);
    Put.of(new File("/home/datahaki/egreedyc"), AllPairs.performance(list, 20, 500));
    System.out.println("done.");
  }
}
