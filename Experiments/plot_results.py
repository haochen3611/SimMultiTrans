import SimMultiTrans as smt

graph = smt.default_graph()
plt = smt.Plot(graph=graph)
plt.import_results()

plt.plot_passenger_queuelen_time(mode='taxi')
plt.plot_passenger_waittime(mode='taxi')
plt.plot_metrics(mode='taxi')

